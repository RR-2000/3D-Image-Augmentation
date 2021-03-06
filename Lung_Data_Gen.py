import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



path_to_npy = './conversion_sample/'
path_to_out = './Generated Data/'

df = pd.read_csv("./characteristics_2.csv")
df = df.drop(['SessionID','Radiologist','Subtlety','internalStructure',\
'Calcification','Sphericity','Margin','Lobulation','Spiculation','Texture'],axis = 1)

df_final = {}

dfname = df.PatientID.unique()
for i in dfname:
    rslt_df = df[(df['PatientID'] == i) & (df['Malignancy']>= 0)]
    anoclass = 1
    rslt_tmp = rslt_df.mean(axis = 0)
    if(rslt_tmp['Malignancy'] >= 3):
        anoclass = 2
    df_final[i] = anoclass


image_names = []
mask_names = []

for i,(PatientID, Class) in enumerate(df_final.items()):
    print('Processing File {} / {}'.format(i+1, len(df_final)), end = "\r")

    if Class == 2:
        continue

    scan = np.load(path_to_npy + PatientID + '_img.npy')
    mask = np.load(path_to_npy + PatientID + '_rois.npy')

    masked_slices = []

    for x in range(mask.shape[0]):
        if(np.max(mask[x]) > 0):
            masked_slices.append(x)

    Z_offset = len(masked_slices)
    if(Z_offset == 0):
        continue
    if(masked_slices[0]-2*Z_offset < scan.shape[0]):
        mass = mask[masked_slices[0]:masked_slices[0]+Z_offset , :, :] *scan[masked_slices[0]:masked_slices[0]+Z_offset , :, :]

        gen_base_up = scan[masked_slices[0]-2*Z_offset: masked_slices[0]-Z_offset , :, :]*(mask[masked_slices[0]:masked_slices[0]+Z_offset , :, :] < 1)
        gen_up = gen_base_up + mass


        for i in range(len(masked_slices)):
            image_name = PatientID[:-1] + '_' + str(ord(PatientID[-1])-ord('a')) + '_' + str(i) + '_UP_IM' + '.npy'
            mask_name = PatientID[:-1] + '_' + str(ord(PatientID[-1])-ord('a')) + '_' + str(i) + '_UP_MA' + '.npy'
            mask_names.append(mask_name)
            image_names.append(image_name)

            np.save(path_to_out+image_name, gen_up)
            np.save(path_to_out+mask_name, mask[masked_slices[0]:masked_slices[0]+Z_offset , :, :])

    if(masked_slices[0]+2*Z_offset < scan.shape[0]):
        gen_base_down = scan[masked_slices[0]+Z_offset: masked_slices[0]+2*Z_offset , :, :]*(mask[masked_slices[0]:masked_slices[0]+Z_offset , :, :] < 1)
        gen_down = gen_base_down + mass

        for i in range(len(masked_slices)):
            image_name = PatientID[:-1] + '_' + str(ord(PatientID[-1])-ord('a')) + '_' + str(i) + '_DN_IM' + '.npy'
            mask_name = PatientID[:-1] + '_' + str(ord(PatientID[-1])-ord('a')) + '_' + str(i) + '_DN_MA' + '.npy'
            mask_names.append(mask_name)
            image_names.append(image_name)

            np.save(path_to_out+image_name, gen_up)
            np.save(path_to_out+mask_name, mask[masked_slices[0]:masked_slices[0]+Z_offset , :, :])

train_im, test_im, train_ma, test_ma =  train_test_split(image_names, mask_names, test_size=0.2)
val_im, test_im, val_ma, test_ma =  train_test_split(test_im, test_ma , test_size=0.5)

new_df = pd.DataFrame({'Image': train_im, 'Mask':  train_ma})
new_df.to_csv('./gen_meta_train.csv', index = False)

new_df = pd.DataFrame({'Image': test_im, 'Mask':  test_ma})
new_df.to_csv('./gen_meta_test.csv', index = False)

new_df = pd.DataFrame({'Image': val_im, 'Mask':  val_ma})
new_df.to_csv('./gen_meta_val.csv', index = False)
