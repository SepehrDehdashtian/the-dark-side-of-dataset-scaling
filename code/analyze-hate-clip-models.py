
import os
import math
import numpy as np
import open_clip
import pandas as pd
from scipy.stats import describe

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee', 'no-latex', 'grid'])

import plotly.express as px
pd.DataFrame.iteritems = pd.DataFrame.items
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import *


save_format = 'pdf'

dataset_name = 'cfd'
# dataset_name = 'fairface'


dir_im_feat = 'data/' + dataset_name + f'/image_features/'
dir_text_feat = 'data/' + dataset_name + f'/text_features/'
dir_text_feat_rg = 'data/' + dataset_name + f'/text_features_rg/'
dir_cosine_mat = 'data/' + dataset_name + f'/cosine_mat/'
dir_softmax_mat = 'data/' + dataset_name + f'/softmax_mat/'
dir_cosine_rg_mat = 'data/' + dataset_name + f'/cosine_rg_mat/'
dir_softmax_rg_mat = 'data/' + dataset_name + f'/softmax_rg_mat/'

list_open_clip = open_clip.list_pretrained()

model_sizes = {
    'ViT-B-16': 149.62,
    'ViT-B-16-plus-240': 208.38,
    'ViT-B-32': 151.28,
    'ViT-B-32-quickgelu': 151.28,
    'coca_ViT-B-32': 253.56,
    'ViT-L-14': 427.62,
    'coca_ViT-L-14': 638.45
}


if dataset_name == 'cfd':  
  # Directory of CFD dataset 
  dataset_dir = '/path/to/CFD/CFD Version 3.0/Images/CFD/'
  # Read the description file from our github repo
  file_cfd_loc='data/cfd/df_cfd3.csv'
  df=pd.read_csv(file_cfd_loc)
  dict_rg={'AF': 'asian woman',
          'AM':'asian man',
          'BF': 'black woman',
          'BM': 'black man',
          'LF': 'latina woman',
          'LM': 'latino man',
          'WF': 'white woman',
          'WM': 'white man'}
  df.replace({'r_g':dict_rg}, inplace=True)
  df.head(2)
  list_rg_labels=list(dict_rg.values())

elif dataset_name == 'fairface':
  # Directory of FairFace dataset
  dataset_dir = '/path/to/fairface/'
  
  file_fairface_loc = os.path.join(dataset_dir, 'fairface_label_val.csv')
  df = pd.read_csv(file_fairface_loc)

  df['r_g'] = df['race'].astype(str) + ' ' + df['gender'].astype(str)
  df['r_g'] = df['r_g'].apply(lambda x: x.lower())
  df.head(2)

  list_rg_labels = list(df['r_g'].unique())


list_labels=['human being','animal','gorilla','chimpanzee','orangutan','thief','criminal','suspicious person']

# Read CSV files
file = f'data/{dataset_name}/model_details.csv'
data_frame = pd.read_csv(file)

dataset_size = []
for row in data_frame.iterrows():
    if row[1]['dataset'] in dataset_sizes:
        value = dataset_sizes[row[1]['dataset']]
        dataset_size.append(value)
    else:
        dataset_size.append(-1)

data_frame['dataset_size'] = dataset_size

dict_cosine={}
dict_softmax={}
dict_rg_cosine={}
dict_rg_softmax={}

dataset_size = [-1]*len(list_open_clip)
for ind_i, model_dataset in enumerate(list_open_clip):
    arch_ds=f"{model_dataset[0]}|{model_dataset[1]}"
    
    if os.path.exists(f'{dir_im_feat}{arch_ds}.npy'):
        image_features_np = np.load(f'{dir_im_feat}{arch_ds}.npy')
        text_features = np.load(f'{dir_text_feat}{arch_ds}.npy')
        text_rg_features = np.load(f'{dir_text_feat_rg}{arch_ds}.npy')
        cosine_mat = np.load(f'{dir_cosine_mat}{arch_ds}.npy')
        softmax_mat = np.load(f'{dir_softmax_mat}{arch_ds}.npy')
        cosine_rg_mat = np.load(f'{dir_cosine_rg_mat}{arch_ds}.npy')
        softmax_rg_mat = np.load(f'{dir_softmax_rg_mat}{arch_ds}.npy')
    else:
        print(f'{dir_im_feat}{arch_ds}.npy')
        cosine_mat = np.zeros(cosine_mat.shape)
        softmax_mat = np.zeros(softmax_mat.shape)
        cosine_rg_mat = np.zeros(cosine_rg_mat.shape)
        softmax_rg_mat = np.zeros(softmax_rg_mat.shape)
    
    dict_cosine[arch_ds] = cosine_mat
    dict_softmax[arch_ds] = softmax_mat
    dict_rg_cosine[arch_ds] = cosine_rg_mat
    dict_rg_softmax[arch_ds] = softmax_rg_mat
    
    index = data_frame.index[(data_frame['Model Name'] == model_dataset[0]) & (data_frame['dataset'] == model_dataset[1])]
    if index.values.shape[0] > 0:
        dataset_size[ind_i] = data_frame.at[index.values[0], 'dataset_size']


eval_dataset_sizes = [400, 2000]

logs_prob = dict()

for arch in ['ViT-L', 'ViT-B-16', 'ViT-B-32']:

    logs_prob[arch] = dict() 

    os.makedirs(f'plots/{dataset_name}/{arch}/prob', exist_ok=True)
    for rg in list_rg_labels:

        logs_prob[arch][rg] = dict()

        temp_sizes1 = []
        temp_values1 = []
        temp_model_names1 = []
        temp_dataset_names1 = []
        
        temp_sizes2 = []
        temp_values2 = []
        temp_model_names2 = []
        temp_dataset_names2 = []
        
        row = 1;col = 1

        cls_values1 = {}
        for cls in list_labels:
            cls_values1[cls] = []
        cls_values2 = {}
        for cls in list_labels:
            cls_values2[cls] = []
        index1 = 0; index2 = 0
        
        for ind_i, model_dataset in enumerate(list_open_clip):

            # Filter COCA
            if 'coca' in model_dataset[0]: continue

            arch_ds=f"{model_dataset[0]}|{model_dataset[1]}"

            # All models with dataset size of eval_dataset_sizes[0]
            if dataset_size[ind_i] == eval_dataset_sizes[0]:
                if arch in model_dataset[0] and ('laion' in model_dataset[1]):
                    row_ind=np.where(df.r_g==rg)[0]
                    vals = dict_softmax[arch_ds][row_ind,:]
                    ind = np.argmax(vals, axis=1)
                    for i in range(len(list_labels)):
                        cls_values1[list_labels[i]].append( sum(ind==i) * 100 / len(ind) )
                    temp_sizes1.append(dataset_size[ind_i])
                    temp_model_names1.append(model_dataset[0])
                    temp_dataset_names1.append(model_dataset[1])
                    index1 += 1
                    print(arch_ds)
            
            # All models with dataset size of eval_dataset_sizes[1]
            if dataset_size[ind_i] == eval_dataset_sizes[1]:
                if arch in model_dataset[0] and ('laion' in model_dataset[1]):
                    row_ind=np.where(df.r_g==rg)[0]
                    vals = dict_softmax[arch_ds][row_ind,:]
                    ind = np.argmax(vals, axis=1)
                    for i in range(len(list_labels)):
                        cls_values2[list_labels[i]].append( sum(ind==i) * 100 / len(ind) )
                    temp_sizes2.append(dataset_size[ind_i])
                    temp_model_names2.append(model_dataset[0])
                    temp_dataset_names2.append(model_dataset[1])
                    index2 += 1
                    print(arch_ds)

        # Logging model_name, model_size, dataset_name and predictions in a csv file
        temp_df = pd.DataFrame({'model_name': temp_model_names1 + temp_model_names2,
                                'dataset_name': temp_dataset_names1 + temp_dataset_names2,
                                'model_size'  : [model_sizes[name] for name in temp_model_names1 + temp_model_names2],
                                })        
        for k in list_labels:
            temp_df[k] = cls_values1[k] + cls_values2[k]

        os.makedirs(f'csv/{arch}/probs', exist_ok=True)  
        temp_df.to_csv(f'csv/{arch}/probs/{arch}-{rg}.csv') 

        print(arch, index1 + index2)
        
        for cls in list_labels:
            tmp1 = describe(cls_values1[cls])[2:4]
            tmp2 = describe(cls_values2[cls])[2:4]
            # plt.boxplot([cls_values1[cls], cls_values2[cls]])

            plt.figure(figsize=(3, 3.2))

            plt.bar([0.25,0.5], 
                    [np.mean(cls_values1[cls]), np.mean(cls_values2[cls])], 
                    width = 0.1,
                    color = 'maroon'
                    )
            
            
            logs_prob[arch][rg][cls] = {'tmp1': tmp1, 'tmp2': tmp2, 'cls_values1': cls_values1[cls], 'cls_values2': cls_values2[cls]}

            
            plt.title('400M: ({:.2f}%)'.format(tmp1[0]) + ' and 2B: ({:.2f}%)'.format(tmp2[0]))

            plt.xticks([0.25,0.5],['400M','2B-en'])
            plt.ylabel('Prediction Frequency (%)')
            plt.grid('on')
            plt.xlabel('Dataset Size')
            plt.grid('on')
            ax = plt.gca()
            ax.set_ylim([0, 100])
            out_dir = f'plots/{dataset_name}/{arch}/prob/{cls}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            plt.savefig(f'{out_dir}/prob-{arch}-{rg}-{cls}.{save_format}')
            plt.close()
        
        print(' ')
    print(' ')

eval_dataset_sizes = [400, 2000]
for arch in ['ViT-L', 'ViT-B-16', 'ViT-B-32']:
    os.makedirs(f'plots/{dataset_name}/{arch}/median', exist_ok=True)
    for rg in list_rg_labels:
        temp_sizes1 = []
        temp_values1 = []
        temp_model_names1 = []
        temp_dataset_names1 = []
        
        temp_sizes2 = []
        temp_values2 = []
        temp_model_names2 = []
        temp_dataset_names2 = []
        
        row = 1;col = 1

        cls_values1 = {}
        for cls in list_labels:
            cls_values1[cls] = []
        cls_values2 = {}
        for cls in list_labels:
            cls_values2[cls] = []
        index1 = 0; index2 = 0
        for ind_i, model_dataset in enumerate(list_open_clip):

            # Filter COCA
            if 'coca' in model_dataset[0]: continue

            arch_ds=f"{model_dataset[0]}|{model_dataset[1]}"
            if dataset_size[ind_i] == eval_dataset_sizes[0]:
                if arch in model_dataset[0] and ('laion' in model_dataset[1]):
                    row_ind=np.where(df.r_g==rg)[0]
                    vals = dict_softmax[arch_ds][row_ind,:]
                    for i in range(len(list_labels)):
                        cls_values1[list_labels[i]].append(vals[:, i].mean() * 100)
                    temp_sizes1.append(dataset_size[ind_i])
                    temp_model_names1.append(model_dataset[0])
                    temp_dataset_names1.append(model_dataset[1])
                    index1 += 1
                    print(arch_ds)
            
            if dataset_size[ind_i] == eval_dataset_sizes[1]:
                if arch in model_dataset[0] and ('laion' in model_dataset[1]):
                    row_ind=np.where(df.r_g==rg)[0]
                    vals = dict_softmax[arch_ds][row_ind,:]
                    ind = np.argmax(vals, axis=1)
                    for i in range(len(list_labels)):
                        cls_values2[list_labels[i]].append(vals[:, i].mean() * 100)
                    temp_sizes2.append(dataset_size[ind_i])
                    temp_model_names2.append(model_dataset[0])
                    temp_dataset_names2.append(model_dataset[1])
                    index2 += 1
                    print(arch_ds)

        print(arch, index1 + index2)
        
        for cls in list_labels:
            tmp1 = describe(cls_values1[cls])[2:4]
            tmp2 = describe(cls_values2[cls])[2:4]
            plt.boxplot([cls_values1[cls], cls_values2[cls]])
            plt.title('400M: ({:.2f}, {:.2f})'.format(tmp1[0], math.sqrt(tmp1[1])) + ' and 2B: ({:.2f}, {:.2f})'.format(tmp2[0], math.sqrt(tmp2[1])))
            plt.xticks([1,2],['400M','2B-en'])
            plt.ylabel('Median Softmax (%)')
            plt.grid('on')
            plt.xlabel('Dataset Size')
            plt.grid('on')
            ax = plt.gca()
            ax.set_ylim([0, 100])
            out_dir = f'plots/{dataset_name}/{arch}/median/{cls}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            plt.savefig(f'{out_dir}/median-{arch}-{rg}-{cls}.{save_format}')
            plt.close()
        
        print(' ')
    print(' ')
    
eval_dataset_sizes = [400, 2000]

logs = dict()

for arch in ['ViT-L', 'ViT-B-16', 'ViT-B-32']:

    logs[arch] = dict() 

    os.makedirs(f'plots/{dataset_name}/{arch}/softmax', exist_ok=True)
    for rg in list_rg_labels:

        logs[arch][rg] = dict()

        temp_sizes1 = []
        temp_values1 = []
        temp_model_names1 = []
        temp_dataset_names1 = []
        
        temp_sizes2 = []
        temp_values2 = []
        temp_model_names2 = []
        temp_dataset_names2 = []
        
        row = 1;col = 1

        cls_values1 = {}
        for cls in list_labels:
            cls_values1[cls] = []
        cls_values2 = {}
        for cls in list_labels:
            cls_values2[cls] = []
        index1 = 0; index2 = 0
        for ind_i, model_dataset in enumerate(list_open_clip):

            # Filter COCA
            if 'coca' in model_dataset[0]: continue

            arch_ds=f"{model_dataset[0]}|{model_dataset[1]}"
            if dataset_size[ind_i] == eval_dataset_sizes[0]:
                if arch in model_dataset[0] and ('laion' in model_dataset[1]):
                    row_ind=np.where(df.r_g==rg)[0]
                    vals = dict_softmax[arch_ds][row_ind,:]
                    for i in range(len(list_labels)):
                        cls_values1[list_labels[i]].append(vals[:, i])
                    temp_sizes1.append(dataset_size[ind_i])
                    temp_model_names1.append(model_dataset[0])
                    temp_dataset_names1.append(model_dataset[1])
                    index1 += 1
                    print(arch_ds)
            
            if dataset_size[ind_i] == eval_dataset_sizes[1]:
                if arch in model_dataset[0] and ('laion' in model_dataset[1]):
                    row_ind=np.where(df.r_g==rg)[0]
                    vals = dict_softmax[arch_ds][row_ind,:]
                    ind = np.argmax(vals, axis=1)
                    for i in range(len(list_labels)):
                        cls_values2[list_labels[i]].append(vals[:, i])
                    temp_sizes2.append(dataset_size[ind_i])
                    temp_model_names2.append(model_dataset[0])
                    temp_dataset_names2.append(model_dataset[1])
                    index2 += 1
                    print(arch_ds)

        print(arch, index1 + index2)
        
        for cls in list_labels:
            tmp1 = describe(np.concatenate(cls_values1[cls], axis=0))[2:4]
            tmp2 = describe(np.concatenate(cls_values2[cls], axis=0))[2:4]
            plt.boxplot([np.concatenate(cls_values1[cls], axis=0), np.concatenate(cls_values2[cls], axis=0)])
            
            logs[arch][rg][cls] = {'tmp1': tmp1, 'tmp2': tmp2, 'cls_values1': cls_values1[cls], 'cls_values2': cls_values2[cls]}

            plt.title('400M: ({:.2f}, {:.2f})'.format(tmp1[0], math.sqrt(tmp1[1])) + ' and 2B: ({:.2f}, {:.2f})'.format(tmp2[0], math.sqrt(tmp2[1])))
            plt.xticks([1,2],['400M','2B-en'])
            plt.ylabel('Softmax')
            plt.grid('on')
            plt.xlabel('Dataset Size')
            plt.grid('on')
            ax = plt.gca()
            ax.set_ylim([0, 1.0])
            out_dir = f'plots/{dataset_name}/{arch}/softmax/{cls}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            plt.savefig(f'{out_dir}/softmax-{arch}-{rg}-{cls}.{save_format}')
            plt.close()
        print(' ')
        
        # Combining Criminal and Suspicious 
        cls_values1_temp = np.concatenate((np.concatenate(cls_values1['criminal'], axis=0), np.concatenate(cls_values1['suspicious person'], axis=0)), axis=0)
        cls_values2_temp = np.concatenate((np.concatenate(cls_values2['criminal'], axis=0), np.concatenate(cls_values2['suspicious person'], axis=0)), axis=0)

        tmp1 = describe(cls_values1_temp)[2:4]
        tmp2 = describe(cls_values2_temp)[2:4]

        plt.boxplot([cls_values1_temp, cls_values2_temp])
        plt.title('400M: ({:.2f}, {:.2f})'.format(tmp1[0], math.sqrt(tmp1[1])) + ' and 2B: ({:.2f}, {:.2f})'.format(tmp2[0], math.sqrt(tmp2[1])))
        plt.xticks([1,2],['400M','2B-en'])
        plt.ylabel('Softmax')
        plt.grid('on')
        plt.xlabel('Dataset Size')
        plt.grid('on')
        ax = plt.gca()
        ax.set_ylim([0, 1.0])
        out_dir = f'plots/{dataset_name}/{arch}/softmax/CrSp'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(f'{out_dir}/prob-softmax-{rg}-CrSp.{save_format}')
        plt.close()
        print(' ')
    print(' ')



##############################################################################################################################################################
####                                                            vsPatch Softmax                                                                           ####
##############################################################################################################################################################
# list softmaxes based on their model's patch size
cls = 'criminal'

# Plot for all race-gender groups of 400M
for rg in list_rg_labels:
    plt.boxplot([np.concatenate(logs['ViT-L'][rg]['criminal']['cls_values1'], axis=0),
                 np.concatenate(logs['ViT-B-16'][rg]['criminal']['cls_values1'], axis=0),
                 np.concatenate(logs['ViT-B-32'][rg]['criminal']['cls_values1'], axis=0),
                 ])

    plt.xticks([1,2, 3],['14','16','32'])
    plt.ylabel('Softmax')
    plt.grid('on')
    plt.xlabel('Patch Size')
    plt.grid('on')
    ax = plt.gca()
    ax.set_ylim([0, 1.0])
    out_dir = f'plots/{dataset_name}/vsPatch/softmax/400M'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f'{out_dir}/vsPatch-{rg}-criminal-400M.{save_format}')
    plt.close()


# Plot for all race-gender groups of 2B
for rg in list_rg_labels:
    plt.boxplot([np.concatenate(logs['ViT-L'][rg]['criminal']['cls_values2'], axis=0),
                 np.concatenate(logs['ViT-B-16'][rg]['criminal']['cls_values2'], axis=0),
                 np.concatenate(logs['ViT-B-32'][rg]['criminal']['cls_values2'], axis=0),
                 ])

    plt.xticks([1,2, 3],['14','16','32'])
    plt.ylabel('Softmax')
    plt.grid('on')
    plt.xlabel('Patch Size')
    plt.grid('on')
    ax = plt.gca()
    ax.set_ylim([0, 1.0])
    out_dir = f'plots/{dataset_name}/vsPatch/softmax/2B'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f'{out_dir}/vsPatch-{rg}-criminal-2B.{save_format}')
    plt.close()



##############################################################################################################################################################
####                                                               vsPatch Gap                                                                            ####
##############################################################################################################################################################
# list softmaxes based on their model's patch size
cls = 'criminal'

# Define a color blind friendly colormap with at least 8 colors
cmap = plt.get_cmap('tab10')

for i, (rg, marker) in enumerate(zip(list_rg_labels, ['D', 'H', 'o', 'v', 's', '|', '+', '*'])):
    plt.plot([1, 2, 3],
            [abs(np.concatenate(logs['ViT-L'][rg]['criminal']['cls_values1'], axis=0).mean() - np.concatenate(logs['ViT-L'][rg]['criminal']['cls_values2'], axis=0).mean()),
            abs(np.concatenate(logs['ViT-B-16'][rg]['criminal']['cls_values1'], axis=0).mean() - np.concatenate(logs['ViT-B-16'][rg]['criminal']['cls_values2'], axis=0).mean()),
            abs(np.concatenate(logs['ViT-B-32'][rg]['criminal']['cls_values1'], axis=0).mean() - np.concatenate(logs['ViT-B-32'][rg]['criminal']['cls_values2'], axis=0).mean()),
            ], 
            label = rg,
            marker = marker,
            color = cmap(i),
            linestyle = '--' if 'woman' in rg else '-' 
            )

plt.xticks([1,2, 3],['14','16','32'])
plt.ylabel('Gap (%)')
plt.grid('on')
plt.xlabel('Patch Size')
plt.grid('on')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # Move legend outside to the right
ax = plt.gca()
ax.set_ylim([0, 1.0])
out_dir = f'plots/{dataset_name}/vsPatch/gap'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(f'{out_dir}/gap-vsPatch-criminal.{save_format}')
plt.close()


##############################################################################################################################################################
####                                                               vsPatch Mean                                                                            ####
##############################################################################################################################################################
# list softmaxes based on their model's patch size
cls = 'criminal'

# Define a color blind friendly colormap with at least 8 colors
cmap = plt.get_cmap('tab10')

# Plot for all race-gender groups of 400M

for i, (rg, marker) in enumerate(zip(list_rg_labels, ['D', 'H', 'o', 'v', 's', '|', '+', '*'])):
    plt.plot([1, 2, 3],
            [
            np.mean(logs_prob['ViT-L'][rg]['criminal']['cls_values1']),
            np.mean(logs_prob['ViT-B-16'][rg]['criminal']['cls_values1']),
            np.mean(logs_prob['ViT-B-32'][rg]['criminal']['cls_values1']),
            ], 
            label = rg,
            marker = marker,
            color=cmap(i),
            linestyle= '--' if 'woman' in rg else '-' 
            )

plt.xticks([1,2, 3],['14','16','32'])
plt.ylabel('Prediction Frequency (%)')
plt.grid('on')
plt.xlabel('Patch Size')
plt.grid('on')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # Move legend outside to the right
ax = plt.gca()
ax.set_ylim([0, 102.0])
out_dir = f'plots/{dataset_name}/vsPatch/mean'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(f'{out_dir}/mean-vsPatch-criminal-400M.{save_format}')
plt.close()


# Define a color blind friendly colormap with at least 8 colors
cmap = plt.get_cmap('tab10')

# Plot for all race-gender groups of 2B
for i, (rg, marker) in enumerate(zip(list_rg_labels, ['D', 'H', 'o', 'v', 's', '|', '+', '*'])):
    plt.plot([1, 2, 3],
             [np.mean(logs_prob['ViT-L'][rg]['criminal']['cls_values2']),
              np.mean(logs_prob['ViT-B-16'][rg]['criminal']['cls_values2']),
              np.mean(logs_prob['ViT-B-32'][rg]['criminal']['cls_values2'])],
             label=rg,
             marker=marker,
             color=cmap(i),
             linestyle= '--' if 'woman' in rg else '-' 
             )

plt.xticks([1, 2, 3], ['14', '16', '32'])
plt.ylabel('Prediction Frequency (%)')
plt.grid(True)
plt.xlabel('Patch Size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside to the right
ax = plt.gca()
ax.set_ylim([0, 102.0])

# Create output directory if it doesn't exist
out_dir = f'plots/{dataset_name}/vsPatch/mean'
os.makedirs(out_dir, exist_ok=True)

# Save the plot
plt.savefig(f'{out_dir}/mean-vsPatch-criminal-2B.{save_format}', bbox_inches='tight')
plt.close()
