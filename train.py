import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import AverageMeter, getMetaDataTask
import glob
from utils import evaluate as evaluate_spotting
from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate as evaluate_dvc
from nlgeval import NLGEval
from torch.nn.utils.rnn import pack_padded_sequence

import wandb

caption_scorer = NLGEval(no_glove=True, no_skipthoughts=True)

def trainer(phase, train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")

    best_loss = 9e99

    os.makedirs(os.path.join("models", model_name, phase), exist_ok=True)
    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, phase, "model.pth.tar")

        # train for one epoch
        loss_training = train(phase, train_loader, model, criterion,
                              optimizer, epoch + 1, train=True)

        # evaluate on validation set
        loss_validation = train(phase, val_loader, model, criterion, optimizer, epoch + 1, train=False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            if phase == "caption":
                test = validate_captioning
            elif phase == "spotting":
                test = validate_spotting
            elif phase == "classifying":
                test = validate_classifying
            # test = validate_captioning if phase == "caption" else validate_spotting
            performance_validation = test(
                val_metric_loader,
                model,
                model_name)

            logging.info("Validation performance at epoch " +
                         str(epoch+1) + " -> " + str(performance_validation))
            
            wandb.log({**{
                f"loss_train_{phase}": loss_training,
                f"loss_val_{phase}": loss_validation,
                "epoch" : epoch,
                }, **{f"{k}_val" : v for k, v in performance_validation.items()}} )
            torch.save(state, os.path.join("models", model_name, phase, f"model_{epoch}.pth.tar"))
        else:
            wandb.log({
                f"loss_train_{phase}": loss_training,
                f"loss_val_{phase}": loss_validation,
                "epoch" : epoch,
                })

        # Reduce LR on Plateau after patience reached
        # prevLR = optimizer.param_groups[0]['lr']
        # scheduler.step(loss_validation)
        # currLR = optimizer.param_groups[0]['lr']
        # if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
        #     logging.info("Plateau Reached!")

        # if (prevLR < 2 * scheduler.eps and
        #         scheduler.num_bad_epochs >= scheduler.patience):
        #     logging.info(
        #         "Plateau Reached and no more reduction -> Exiting Loop")
        #     break
        scheduler.step()

    return

def train(phase, dataloader, model, criterion, optimizer, epoch, train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, batch in t:
            # measure data loading time
            data_time.update(time.time() - end)
            if phase == "spotting":
                # spotting 单流: (vfeats, labels) 或 双流: (vfeats, afeats, labels)
                if len(batch) == 3 and not isinstance(batch[0], str):
                    # 双流 spotting: batch = (vfeats, afeats, labels)
                    vfeats, afeats, labels = batch
                    vfeats = vfeats.cuda()
                    afeats = afeats.cuda()
                    labels = labels.cuda()
                    labels = labels[:, 1].long().cuda()
                    output = model(vfeats, afeats)
                    feats = vfeats  # 给 losses.update 用
                else:
                    feats, labels = batch
                    feats = feats.cuda()
                    labels = labels.cuda()
                    labels = labels[:, 1].long().cuda()
                    output = model(feats)
                loss = criterion(output, labels)
            elif phase == "caption":
                # caption batch 结构: (data_tuple, lengths, mask, caption_or, cap_id)
                #   单流 data_tuple = (feats, caption)
                #   双流 data_tuple = (vfeats, afeats, caption)
                data_tuple = batch[0]
                lengths = batch[1]
                mask = batch[2]
                caption_or = batch[3]
                cap_id = batch[4]

                if isinstance(data_tuple, (list, tuple)) and len(data_tuple) == 3:
                    # 双流模式
                    vfeats, afeats, caption = data_tuple
                    vfeats = vfeats.cuda()
                    afeats = afeats.cuda()
                    caption = caption.cuda()
                    target = caption[:, 1:]
                    lengths = lengths - 1
                    target = pack_padded_sequence(target, lengths, batch_first=True, enforce_sorted=False)[0]
                    mask = pack_padded_sequence(mask[:, 1:], lengths, batch_first=True, enforce_sorted=False)[0]
                    output = model(vfeats, afeats, caption, lengths)
                    feats = vfeats
                else:
                    # 原有单流模式
                    feats, caption = data_tuple
                    caption = caption.cuda()
                    target = caption[:, 1:]
                    lengths = lengths - 1
                    target = pack_padded_sequence(target, lengths, batch_first=True, enforce_sorted=False)[0]
                    mask = pack_padded_sequence(mask[:, 1:], lengths, batch_first=True, enforce_sorted=False)[0]
                    feats = feats.cuda()
                    output = model(feats, caption, lengths)
                
                loss = criterion(output[mask], target[mask])
            elif phase == "classifying":
                # classifying 单流: (feats, labels) 或 双流: (vfeats, afeats, labels)
                if len(batch) == 3:
                    vfeats, afeats, labels = batch
                    vfeats = vfeats.cuda()
                    afeats = afeats.cuda()
                    labels = labels.cuda()
                    output = model(vfeats, afeats)
                    feats = vfeats
                else:
                    feats, labels = batch
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                loss = criterion(output, labels)
            else:
                NotImplementedError()
            
            # check if the loss is NaN
            if torch.isnan(loss):  
                logging.error("Loss is NaN")
                print(i)
                logging.info(loss)
                #print(feats, caption, output)
                #logging.info(lengths)
                # print model parameter norms
                # for name, param in model.named_parameters():
                #     logging.info(f"{name} - {param.norm().item()}")
                sys.exit()
            # measure accuracy and record loss
            losses.update(loss.detach().item(), feats.size(0))

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                #clip the gradient to avoid exploding gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    return losses.avg

def validate_spotting(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, batch in t:
            # measure data loading time
            data_time.update(time.time() - end)

            if len(batch) == 3:
                # 双流: (vfeats, afeats, labels)
                vfeats, afeats, labels = batch
                vfeats = vfeats.cuda()
                afeats = afeats.cuda()
                output = model(vfeats, afeats)
            else:
                feats, labels = batch
                feats = feats.cuda()
                output = model(feats)
            
            labels = (labels > 0) * 1.0
            all_labels.append(labels.detach().numpy())

            output = torch.nn.functional.softmax(output, dim=1) # Bx18
            output = torch.stack([output[:, 0], 1-output[:, 0]], dim=1)
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (cls): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    AP = []
    for i in range(1, dataloader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    mAP = np.mean(AP)

    return {"mAP-sklearn" : mAP}

def validate_classifying(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    correct_predictions = 0.0
    total_predictions = 0.0
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for i, batch in t:
                # measure data loading time
                data_time.update(time.time() - end)

                if len(batch) == 3:
                    # 双流: (vfeats, afeats, labels)
                    vfeats, afeats, labels = batch
                    vfeats = vfeats.cuda()
                    afeats = afeats.cuda()
                    output = model(vfeats, afeats)
                else:
                    feats, labels = batch
                    feats = feats.cuda()
                    output = model(feats)

                _, predicted = torch.max(output.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted.cpu().detach() == labels).sum().item()


                batch_time.update(time.time() - end)
                end = time.time()

                desc = f'Test (cls): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)

    return {"accuracy" : correct_predictions/total_predictions}

def test_spotting(dataloader, model, model_name, save_predictions=True, NMS_window=30, NMS_threshold=0.5):
    
    split = '_'.join(dataloader.dataset.split)
    output_folder = f"outputs/{split}"
    output_results = os.path.join("models", model_name, output_folder)
    

    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    _, _, _, inv_dict = getMetaDataTask("caption", "SoccerNet", dataloader.dataset.version)

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, batch_data in t:
            data_time.update(time.time() - end)

            # 兼容双流 (7元素) 和单流 (5元素) batch
            if len(batch_data) == 7:
                # 双流: (game_ID, vfeat_h1, vfeat_h2, afeat_h1, afeat_h2, label_h1, label_h2)
                game_ID, feat_half1, feat_half2, audio_half1, audio_half2, label_half1, label_half2 = batch_data
                is_dual = True
            else:
                # 单流: (game_ID, feat_h1, feat_h2, label_h1, label_h2)
                game_ID, feat_half1, feat_half2, label_half1, label_half2 = batch_data
                is_dual = False

            # Batch size of 1
            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.squeeze(0)
            label_half2 = label_half2.float().squeeze(0)
            if is_dual:
                audio_half1 = audio_half1.squeeze(0)
                audio_half2 = audio_half2.squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = []
            for b in range(int(np.ceil(len(feat_half1)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half1) else len(feat_half1)
                feat = feat_half1[start_frame:end_frame].cuda()
                if is_dual:
                    afeat = audio_half1[start_frame:end_frame].cuda()
                    output = model(feat, afeat)
                else:
                    output = model(feat)
                output = torch.nn.functional.softmax(output, dim=1)

                max_ind = torch.argmax(output[:, 1:], dim=1)+1 # Bx1
                output = torch.stack([output[:, 0], output[torch.arange(output.size(0)), max_ind]], dim=1)

                output = output.cpu().detach().numpy()
                timestamp_long_half_1.append(output)
            timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

            timestamp_long_half_2 = []
            for b in range(int(np.ceil(len(feat_half2)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half2) else len(feat_half2)
                feat = feat_half2[start_frame:end_frame].cuda()
                if is_dual:
                    afeat = audio_half2[start_frame:end_frame].cuda()
                    output = model(feat, afeat)
                else:
                    output = model(feat)
                output = torch.nn.functional.softmax(output, dim=1)

                max_ind = torch.argmax(output[:, 1:], dim=1)+1 # Bx1
                output = torch.stack([output[:, 0], output[torch.arange(output.size(0)), max_ind]], dim=1)

                output = output.cpu().detach().numpy()
                timestamp_long_half_2.append(output)
            timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)


            timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
            timestamp_long_half_2 = timestamp_long_half_2[:, 1:] # Bx1 

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)



            def get_spot_from_NMS(Input, window=60, thresh=0.0):

                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index,0))
                    nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1

                return np.transpose([indexes, MaxValues])

            framerate = dataloader.dataset.framerate
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()

            for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(dataloader.dataset.num_classes):
                    spots = get_spot(
                        timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold) # l = 0 which is out[:, 1:][:, 0]
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        # confidence = predictions_half_1[frame_index, l]

                        seconds = int((frame_index//framerate)%60)
                        minutes = int((frame_index//framerate)//60)

                        prediction_data = dict()
                        prediction_data["gameTime"] = f'{half+1} - {int(minutes):02d}:{int(seconds):02d}'
                        prediction_data["label"] = inv_dict[l]

                        prediction_data["position"] = str(int((frame_index/framerate)*1000))
                        prediction_data["half"] = str(half+1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)
            
            json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: (int(x["half"]), int(x["position"])))
            if save_predictions:
                os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    
    tight = evaluate_spotting(SoccerNet_path=dataloader.dataset.path, 
                Predictions_path=output_results,
                split=dataloader.dataset.split,
                prediction_file="results_spotting.json", 
                version=dataloader.dataset.version, 
                framerate=dataloader.dataset.framerate, metric="tight")
    
    loose = evaluate_spotting(SoccerNet_path=dataloader.dataset.path, 
                Predictions_path=output_results,
                split=dataloader.dataset.split,
                prediction_file="results_spotting.json", 
                version=dataloader.dataset.version, 
                framerate=dataloader.dataset.framerate, metric="loose")
    
    medium = evaluate_spotting(SoccerNet_path=dataloader.dataset.path, 
                Predictions_path=output_results,
                split=dataloader.dataset.split,
                prediction_file="results_spotting.json", 
                version=dataloader.dataset.version, 
                framerate=dataloader.dataset.framerate, metric="medium")

    tight = {f"{k}_tight" : v for k, v in tight.items() if v!= None}
    loose = {f"{k}_loose" : v for k, v in loose.items() if v!= None}
    medium = {f"{k}_medium" : v for k, v in medium.items() if v!= None}

    results = {**tight, **loose, **medium}

    return results

@torch.no_grad()
def validate_captioning(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    
    with tqdm(dataloader) as t:
        for batch in t:
            # measure data loading time
            data_time.update(time.time() - end)

            # 兼容双流和单流 batch 格式
            data_tuple = batch[0]
            lengths = batch[1]
            mask = batch[2]
            caption_or = batch[3]
            cap_id = batch[4]

            if isinstance(data_tuple, (list, tuple)) and len(data_tuple) == 3:
                # 双流: (vfeats, afeats, caption)
                vfeats, afeats, caption = data_tuple
                vfeats = vfeats.cuda()
                afeats = afeats.cuda()
                output = [dataloader.dataset.detokenize(list(model.sample(vfeats[idx], afeats[idx]).detach().cpu())) for idx in range(vfeats.shape[0])]
            else:
                # 单流: (feats, caption)
                feats, caption = data_tuple
                feats = feats.cuda()
                output = [dataloader.dataset.detokenize(list(model.sample(feats[idx]).detach().cpu())) for idx in range(feats.shape[0])]
            
            all_outputs.extend(output)
            all_labels.extend(caption_or)
            print("Output:", output[0])
            print("Labels:", caption_or[0])
            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (cap): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    scores = caption_scorer.compute_metrics(ref_list=[all_labels,], hyp_list=all_outputs)
    return scores

@torch.no_grad()
def test_captioning(dataloader, model, model_name, output_filename = "results_dense_captioning.json", input_filename="results_spotting.json"):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_outputs = []
    all_index = []

    split = '_'.join(dataloader.dataset.split)
    output_folder = f"outputs/{split}"
    output_results = os.path.join("models", model_name, f"results_dense_captioning_{split}.zip")

    with tqdm(dataloader) as t:
        for batch in t:
            # measure data loading time
            data_time.update(time.time() - end)

            # 兼容双流和单流: 双流返回 (vfeats, afeats, game_id, cap_id), 单流返回 (feats, game_id, cap_id)
            if len(batch) == 4:
                # 双流 PredictionCaptionsDual
                vfeats, afeats, game_id, cap_id = batch
                vfeats = vfeats.cuda()
                afeats = afeats.cuda()
                output = [dataloader.dataset.detokenize(list(model.sample(vfeats[idx], afeats[idx]).detach().cpu())) for idx in range(vfeats.shape[0])]
            else:
                # 单流 PredictionCaptions
                feats, game_id, cap_id = batch
                feats = feats.cuda()
                output = [dataloader.dataset.detokenize(list(model.sample(feats[idx]).detach().cpu())) for idx in range(feats.shape[0])]
            
            all_outputs.extend(output)
            all_index.extend([(i.item(), j.item()) for i, j in zip(game_id, cap_id)])

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (dense_caption): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)
    
    #store output
    captions = dict(zip(all_index, all_outputs))
    skipped_games = 0
    for game_id, game in enumerate(dataloader.dataset.listGames):
        path = os.path.join("models", model_name, output_folder, game, input_filename)
        with open(path, 'r') as pred_file:
            preds = json.load(pred_file)
        # Skip games with no spotting predictions to avoid UnboundLocalError
        # in SoccerNet evaluator (evaluate_detection iterates over empty list)
        if len(preds["predictions"]) == 0:
            logging.warning(f"Game '{game}' has 0 spotting predictions - inserting a dummy prediction to avoid evaluation crash.")
            skipped_games += 1
            # Insert a harmless dummy prediction to bypass the SoccerNet library bug
            preds["predictions"].append({
                "gameTime": "1 - 00:00",
                "label": "Corner",
                "position": "0",
                "half": "1",
                "confidence": "0.0",
                "comment": ""
            })
            
        for caption_id, annotation in enumerate(preds["predictions"]):
            annotation["comment"] = captions.get((game_id, caption_id), "")
        with open(os.path.join("models", model_name, output_folder, game, output_filename), 'w') as output_file:
            json.dump(preds, output_file, indent=4)
    if skipped_games > 0:
        logging.warning(f"Skipped {skipped_games} games with empty predictions during captioning output.")
    
    def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            rootlen = len(target_dir) + 1
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipobj:
                for base, dirs, files in os.walk(target_dir):
                    for file in files:
                        if file == filename:
                            fn = os.path.join(base, file)
                            # 统一使用 / 作为 zip 内部路径分隔符，避免 Windows 平台问题
                            arcname = fn[rootlen:].replace(os.sep, '/')
                            zipobj.write(fn, arcname)
    
    zipResults(zip_path = output_results,
            target_dir = os.path.join("models", model_name, output_folder),
            filename=output_filename)

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    
    tight = evaluate_dvc(SoccerNet_path=dataloader.dataset.path, Predictions_path=output_results, split=dataloader.dataset.split, version=dataloader.dataset.version, prediction_file=output_filename, window_size=5, include_SODA=False)
    loose = evaluate_dvc(SoccerNet_path=dataloader.dataset.path, Predictions_path=output_results, split=dataloader.dataset.split, version=dataloader.dataset.version, prediction_file=output_filename, window_size=30, include_SODA=False)

    results = {**{f"{k}_tight" : v for k, v in tight.items()}, **{f"{k}_loose" : v for k, v in loose.items()}}

    return results