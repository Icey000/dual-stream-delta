#!/bin/bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=80G --nodelist=n2 --time=10:00:00 --pty bash