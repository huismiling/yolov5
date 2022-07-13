import os
import re
import sys
import argparse
import shutil

import torch
import torch_mlu
import torch_mlu

parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    '-i',
                    default='./',
                    required=True,
                    type=str,
                    help='input file.')
parser.add_argument('--strict',
                    '-s',
                    action='store_true',
                    help='strict conversion mode.')

args = parser.parse_args()

input_path = os.path.abspath(args.input)
mlu_path = input_path + '_mlu'

if not os.path.exists(mlu_path):
    os.mkdir(mlu_path)

mlu_report = os.path.join(mlu_path, 'report.md')
report = open(mlu_report, 'w+')

report.write('# Cambricon PyTorch Model Migration Report\n')
report.write('## Cambricon PyTorch Changes\n')
report.write('| No. |  File  |  Description  |\n')

num = 0
for root, dirs, files in os.walk(input_path):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = file_path[len(input_path) + 1:]
        mlu_file_path = mlu_path + file_path[len(input_path):]
        root_mlu = os.path.dirname(mlu_file_path)
        if not os.path.exists(root_mlu):
            os.makedirs(root_mlu)

        if not file_path.endswith(".py"):
            try:
                shutil.copy(file_path, mlu_file_path)
            except:
                print('copy error: ', file_path)
        else:
            f = open(file_path, 'r+')
            mlu_f = open(mlu_file_path, 'w+')
            line = 0
            has_import = False
            for ss in f.readlines():
                line = line + 1
                if ss.strip() == 'import torch' and not has_import:
                    num = num + 1
                    has_import = True
                    mlu_f.write(ss)
                    ss = re.sub('torch', 'torch_mlu', ss)
                    mlu_f.write(ss)
                    report.write('| ' + str(num) + ' | ' + relative_path +
                                 ':' + str(line) +
                                 ' | add \"import torch_mlu\" |\n')
                    continue

                ori_ss = ss
                ss = re.sub('mlu', 'mlu', ss)
                ss = re.sub('cncl', 'cncl', ss)

                if args.strict:
                    ss = re.sub('CUDA', 'MLU', ss)
                    ss = re.sub('gpu', 'mlu', ss)
                    ss = re.sub('GPU', 'MLU', ss)

                mlu_f.write(ss)
                if ori_ss != ss:
                    num = num + 1
                    report.write('| ' + str(num) + ' | ' + relative_path +
                                 ':' + str(line) + ' | change \"' +
                                 ori_ss.strip() + '\" to \"' + ss.strip() +
                                 ' \" |\n')

print('# Cambricon PyTorch Model Migration Report')
print('Official PyTorch model scripts: ', input_path)
print('Cambricon PyTorch model scripts: ', mlu_path)
print('Migration Report: ', mlu_report)
