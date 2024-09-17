import argparse
import glob,shutil, os,random
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image






def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["Kvasir", "CVC-ColonDB", "CVC-ClinicDB", "ETIS", "CVC-300", "PolypGen", "SUN-SEG"], default="PolypGen",
                        help="Which downstream task.")
    args = parser.parse_args()



    

################################################################################################################################
    if args.dataset == "Kvasir" :

        !mkdir -p "./Kvasir/train/images"
        !mkdir -p "./Kvasir/train/labels"
        !mkdir -p "./Kvasir/valid/images"
        !mkdir -p "./Kvasir/valid/labels"
        
        def csv_txt(dfs, target_size):
          x,y=target_size
          list=[]
          for w in range(len(dfs)):
            row = dfs.iloc[w]
            x_c = (row['xmax']+row['xmin'])/2/x
            y_c = (row['ymax']+row['ymin'])/2/y
            w = (row['xmax']-row['xmin'])/x
            h = (row['ymax']-row['ymin'])/y
            list.append([0,x_c,y_c,w,h])
          return list
        
        output_dir_label ="./Kvasir/train/labels"
        output_dir_image = "./Kvasir/train/images"
        image_list = glob.glob("./Kvasir-SEG/Kvasir-SEG/images/*.jpg")
        for i in range(800):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./Kvasir-SEG/Kvasir-SEG/bbox/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)
        
        
        
        output_dir_label ="./Kvasir/valid/labels"
        output_dir_image = "./Kvasir/valid/images"
        for i in range(800,len(image_list)):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./Kvasir-SEG/Kvasir-SEG/bbox/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)



################################################################################################################################
    elif  args.dataset == "ColonDB" :

        class CVC_colondb(torch.utils.data.Dataset):
        
            def __init__(self, n_points_add, n_points_rmv, transform=None):
                self.root_dir = glob.glob("./CVC-ColonDB/images/*.png")
                self.n_points_add = n_points_add
                self.n_points_rmv = n_points_rmv
                self.transform = transform
        
            def __len__(self):
                return len(self.root_dir)
        
            def __getitem__(self, idx):
                img_pth = self.root_dir[idx]
                msk_pth = './CVC-ColonDB/masks/' + img_pth.split('/')[-1]
        
                image = Image.open(img_pth)
                image = np.array(image.convert("RGB"))
        
                if not os.path.exists(msk_pth):
                    raise FileNotFoundError(f"Mask file not found: {msk_pth}")
        
                mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                    raise ValueError(f"Failed to read mask image: {msk_pth}")
                mask_np = mask_np / 255
        
                if self.transform:
                    image = self.transform(image)
        
                seg_value = 1.
                segmentation = np.where(mask_np == seg_value)
        
                bboxes = 0, 0, 0, 0
                if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                    x_min = int(np.min(segmentation[1]))
                    x_max = int(np.max(segmentation[1]))
                    y_min = int(np.min(segmentation[0]))
                    y_max = int(np.max(segmentation[0]))
        
                    bboxes = x_min, y_min, x_max, y_max
        
                    num_masks = 1
        
                point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
                point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
                bounding_box = np.zeros((num_masks, 4))
        
                ori_add = np.where(mask_np == 1)
                ori_rmv = np.where(mask_np == 0)
        
                if ori_add[0].shape[0] < self.n_points_add or ori_rmv[0].shape[0] < self.n_points_rmv:
                    raise ValueError("Not enough points to sample from.")
        
                rand_add = np.random.randint(ori_add[0].shape[0], size=self.n_points_add)
                rand_rmv = np.random.randint(ori_rmv[0].shape[0], size=self.n_points_rmv)
        
                for i in range(self.n_points_add):
                    point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
                    point_labels[0, i] = 1
        
                for i in range(self.n_points_rmv):
                    point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
                    point_labels[0, i + self.n_points_add] = 0
        
                bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        
                return {
                    "image": image,
                    "mask_np": mask_np,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "bounding_box": bounding_box,
                    "image_name": img_pth.split('/')[-1]
                }
        
        
        dataset = CVC_colondb(n_points_add=10, n_points_rmv=10)
        
        
        os.makedirs('labels', exist_ok=True)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            image_name = sample["image_name"]
            x_min, y_min, x_max, y_max = sample["bounding_box"][0]
        
        
            df = pd.DataFrame([[x_min, x_max, y_min, y_max]], columns=["x_min", "x_max", "y_min", "y_max"])
        
        
            csv_filename = os.path.join('labels', f"{image_name.split('.')[0]}.csv")
            df.to_csv(csv_filename, index=False)
        
        print("CSV files saved successfully in the 'labels' folder.")
        
        !mkdir -p "./CVC_colondb/train/images"
        !mkdir -p "./CVC_colondb/train/labels"
        !mkdir -p "./CVC_colondb/valid/images"
        !mkdir -p "./CVC_colondb/valid/labels"
        
        def csv_txt(dfs, target_size):
          x,y=target_size
          list=[]
          for w in range(len(dfs)):
            row = dfs.iloc[w]
            x_c = (row['x_max']+row['x_min'])/2/x
            y_c = (row['y_max']+row['y_min'])/2/y
            w = (row['x_max']-row['x_min'])/x
            h = (row['y_max']-row['y_min'])/y
            list.append([0,x_c,y_c,w,h])
          return list
        
        output_dir_label ="./CVC_colondb/train/labels/"
        output_dir_image = "./CVC_colondb/train/images/"
        image_list = glob.glob("./CVC-ColonDB/images/*.png")
        for i in range(304):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)
        
          output_dir_label ="./CVC_colondb/valid/labels/"
        output_dir_image = "./CVC_colondb/valid/images/"
        for i in range(304,len(image_list)):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)



################################################################################################################################
    elif  args.dataset == "CVC-ClinicDB" :

        class CVC_clinicdb(torch.utils.data.Dataset):
        
            def __init__(self, n_points_add, n_points_rmv, transform=None):
                self.root_dir = glob.glob("./PNG/Original/*.png")
                self.n_points_add = n_points_add
                self.n_points_rmv = n_points_rmv
                self.transform = transform
        
            def __len__(self):
                return len(self.root_dir)
        
            def __getitem__(self, idx):
                img_pth = self.root_dir[idx]
                msk_pth = './PNG/Ground Truth/' + img_pth.split('/')[-1]
        
                image = Image.open(img_pth)
                image = np.array(image.convert("RGB"))
        
                if not os.path.exists(msk_pth):
                    raise FileNotFoundError(f"Mask file not found: {msk_pth}")
        
                mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                    raise ValueError(f"Failed to read mask image: {msk_pth}")
                mask_np = mask_np / 255
        
                if self.transform:
                    image = self.transform(image)
        
                seg_value = 1.
                segmentation = np.where(mask_np == seg_value)
        
                bboxes = 0, 0, 0, 0
                if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                    x_min = int(np.min(segmentation[1]))
                    x_max = int(np.max(segmentation[1]))
                    y_min = int(np.min(segmentation[0]))
                    y_max = int(np.max(segmentation[0]))
        
                    bboxes = x_min, y_min, x_max, y_max
        
                    num_masks = 1
        
                point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
                point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
                bounding_box = np.zeros((num_masks, 4))
        
                ori_add = np.where(mask_np == 1)
                ori_rmv = np.where(mask_np == 0)
        
                if ori_add[0].shape[0] < self.n_points_add or ori_rmv[0].shape[0] < self.n_points_rmv:
                    raise ValueError("Not enough points to sample from.")
        
                rand_add = np.random.randint(ori_add[0].shape[0], size=self.n_points_add)
                rand_rmv = np.random.randint(ori_rmv[0].shape[0], size=self.n_points_rmv)
        
                for i in range(self.n_points_add):
                    point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
                    point_labels[0, i] = 1
        
                for i in range(self.n_points_rmv):
                    point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
                    point_labels[0, i + self.n_points_add] = 0
        
                bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        
                return {
                    "image": image,
                    "mask_np": mask_np,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "bounding_box": bounding_box,
                    "image_name": img_pth.split('/')[-1]
                }
        
        
        dataset = CVC_clinicdb(n_points_add=10, n_points_rmv=10)
        
        os.makedirs('labels', exist_ok=True)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            image_name = sample["image_name"]
            x_min, y_min, x_max, y_max = sample["bounding_box"][0]
        
            df = pd.DataFrame([[x_min, x_max, y_min, y_max]], columns=["x_min", "x_max", "y_min", "y_max"])
        
            csv_filename = os.path.join('labels', f"{image_name.split('.')[0]}.csv")
            df.to_csv(csv_filename, index=False)
        
        print("CSV files saved successfully in the 'labels' folder.")
        
        !mkdir -p "./CVC_clinicdb/train/images"
        !mkdir -p "./CVC_clinicdb/train/labels"
        !mkdir -p "./CVC_clinicdb/valid/images"
        !mkdir -p "./CVC_clinicdb/valid/labels"
        
        def csv_txt(dfs, target_size):
          x,y=target_size
          list=[]
          for w in range(len(dfs)):
            row = dfs.iloc[w]
            x_c = (row['x_max']+row['x_min'])/2/x
            y_c = (row['y_max']+row['y_min'])/2/y
            w = (row['x_max']-row['x_min'])/x
            h = (row['y_max']-row['y_min'])/y
            list.append([0,x_c,y_c,w,h])
          return list
        
        output_dir_label ="./CVC_clinicdb/train/labels/"
        output_dir_image = "./CVC_clinicdb/train/images/"
        image_list = glob.glob("./PNG/Original/*.png")
        for i in range(490):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)
        
        
        output_dir_label ="./CVC_clinicdb/valid/labels/"
        output_dir_image = "./CVC_clinicdb/valid/images/"
        for i in range(490,len(image_list)):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)


        
################################################################################################################################
    elif  args.dataset == "ETIS" :
   
        class ETIS(torch.utils.data.Dataset):
        
            def __init__(self, n_points_add, n_points_rmv, transform=None):
                self.root_dir = glob.glob("./images/*.png")
                self.n_points_add = n_points_add
                self.n_points_rmv = n_points_rmv
                self.transform = transform
        
            def __len__(self):
                return len(self.root_dir)
        
            def __getitem__(self, idx):
                img_pth = self.root_dir[idx]
                msk_pth = './masks/' + img_pth.split('/')[-1]
        
                image = Image.open(img_pth)
                image = np.array(image.convert("RGB"))
        
                if not os.path.exists(msk_pth):
                    raise FileNotFoundError(f"Mask file not found: {msk_pth}")
        
                mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                    raise ValueError(f"Failed to read mask image: {msk_pth}")
                mask_np = mask_np / 255
        
                if self.transform:
                    image = self.transform(image)
        
                seg_value = 1.
                segmentation = np.where(mask_np == seg_value)
        
                bboxes = 0, 0, 0, 0
                if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                    x_min = int(np.min(segmentation[1]))
                    x_max = int(np.max(segmentation[1]))
                    y_min = int(np.min(segmentation[0]))
                    y_max = int(np.max(segmentation[0]))
        
                    bboxes = x_min, y_min, x_max, y_max
        
                    num_masks = 1
        
                point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
                point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
                bounding_box = np.zeros((num_masks, 4))
        
                ori_add = np.where(mask_np == 1)
                ori_rmv = np.where(mask_np == 0)
        
                if ori_add[0].shape[0] < self.n_points_add or ori_rmv[0].shape[0] < self.n_points_rmv:
                    raise ValueError("Not enough points to sample from.")
        
                rand_add = np.random.randint(ori_add[0].shape[0], size=self.n_points_add)
                rand_rmv = np.random.randint(ori_rmv[0].shape[0], size=self.n_points_rmv)
        
                for i in range(self.n_points_add):
                    point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
                    point_labels[0, i] = 1
        
                for i in range(self.n_points_rmv):
                    point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
                    point_labels[0, i + self.n_points_add] = 0
        
                bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        
                return {
                    "image": image,
                    "mask_np": mask_np,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "bounding_box": bounding_box,
                    "image_name": img_pth.split('/')[-1]
                }
        
        
        dataset = CVC_clinicdb(n_points_add=10, n_points_rmv=10)
        
        os.makedirs('labels', exist_ok=True)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            image_name = sample["image_name"]
            x_min, y_min, x_max, y_max = sample["bounding_box"][0]
        
            df = pd.DataFrame([[x_min, x_max, y_min, y_max]], columns=["x_min", "x_max", "y_min", "y_max"])
        
            csv_filename = os.path.join('labels', f"{image_name.split('.')[0]}.csv")
            df.to_csv(csv_filename, index=False)
        
        print("CSV files saved successfully in the 'labels' folder.")
        
        !mkdir -p "./ETIS/train/images"
        !mkdir -p "./ETIS/train/labels"
        !mkdir -p "./ETIS/valid/images"
        !mkdir -p "./ETIS/valid/labels"
        
        def csv_txt(dfs, target_size):
          x,y=target_size
          list=[]
          for w in range(len(dfs)):
            row = dfs.iloc[w]
            x_c = (row['x_max']+row['x_min'])/2/x
            y_c = (row['y_max']+row['y_min'])/2/y
            w = (row['x_max']-row['x_min'])/x
            h = (row['y_max']-row['y_min'])/y
            list.append([0,x_c,y_c,w,h])
          return list
        
        output_dir_label ="./ETIS/train/labels/"
        output_dir_image = "./ETIS/train/images/"
        image_list = glob.glob("./images/*.png")
        for i in range(157):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)
        
          output_dir_label ="./ETIS/valid/labels/"
        output_dir_image = "./ETIS/valid/images/"
        for i in range(157,len(image_list)):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)   
    
    
    
    

################################################################################################################################    
    elif  args.dataset == "CVC-300" :

        class CVC_300(torch.utils.data.Dataset):
        
            def __init__(self, n_points_add, n_points_rmv, transform=None):
                self.root_dir = glob.glob("./CVC-300/images/*.png")
                self.n_points_add = n_points_add
                self.n_points_rmv = n_points_rmv
                self.transform = transform
        
            def __len__(self):
                return len(self.root_dir)
        
            def __getitem__(self, idx):
                img_pth = self.root_dir[idx]
                msk_pth = './CVC-300/masks/' + img_pth.split('/')[-1]
        
                image = Image.open(img_pth)
                image = np.array(image.convert("RGB"))
        
                if not os.path.exists(msk_pth):
                    raise FileNotFoundError(f"Mask file not found: {msk_pth}")
        
                mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                    raise ValueError(f"Failed to read mask image: {msk_pth}")
                mask_np = mask_np / 255
        
                if self.transform:
                    image = self.transform(image)
        
                seg_value = 1.
                segmentation = np.where(mask_np == seg_value)
        
                bboxes = 0, 0, 0, 0
                if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                    x_min = int(np.min(segmentation[1]))
                    x_max = int(np.max(segmentation[1]))
                    y_min = int(np.min(segmentation[0]))
                    y_max = int(np.max(segmentation[0]))
        
                    bboxes = x_min, y_min, x_max, y_max
        
                    num_masks = 1
        
                point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
                point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
                bounding_box = np.zeros((num_masks, 4))
        
                ori_add = np.where(mask_np == 1)
                ori_rmv = np.where(mask_np == 0)
        
                if ori_add[0].shape[0] < self.n_points_add or ori_rmv[0].shape[0] < self.n_points_rmv:
                    raise ValueError("Not enough points to sample from.")
        
                rand_add = np.random.randint(ori_add[0].shape[0], size=self.n_points_add)
                rand_rmv = np.random.randint(ori_rmv[0].shape[0], size=self.n_points_rmv)
        
                for i in range(self.n_points_add):
                    point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
                    point_labels[0, i] = 1
        
                for i in range(self.n_points_rmv):
                    point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
                    point_labels[0, i + self.n_points_add] = 0
        
                bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        
                return {
                    "image": image,
                    "mask_np": mask_np,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "bounding_box": bounding_box,
                    "image_name": img_pth.split('/')[-1]
                }
        
        dataset = CVC_300(n_points_add=10, n_points_rmv=10)
        os.makedirs('labels', exist_ok=True)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            image_name = sample["image_name"]
            x_min, y_min, x_max, y_max = sample["bounding_box"][0]
        
            df = pd.DataFrame([[x_min, x_max, y_min, y_max]], columns=["x_min", "x_max", "y_min", "y_max"])
        
            csv_filename = os.path.join('labels', f"{image_name.split('.')[0]}.csv")
            df.to_csv(csv_filename, index=False)
        
        print("CSV files saved successfully in the 'labels' folder.")
        
        !mkdir -p "./CVC_300/train/images"
        !mkdir -p "./CVC_300/train/labels"
        !mkdir -p "./CVC_300/valid/images"
        !mkdir -p "./CVC_300/valid/labels"
        
        def csv_txt(dfs, target_size):
          x,y=target_size
          list=[]
          for w in range(len(dfs)):
            row = dfs.iloc[w]
            x_c = (row['x_max']+row['x_min'])/2/x
            y_c = (row['y_max']+row['y_min'])/2/y
            w = (row['x_max']-row['x_min'])/x
            h = (row['y_max']-row['y_min'])/y
            list.append([0,x_c,y_c,w,h])
          return list
        
        output_dir_label ="./CVC_300/train/labels/"
        output_dir_image = "./CVC_300/train/images/"
        image_list = glob.glob("./CVC-300/images/*.png")
        for i in range(48):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)
        
          output_dir_label ="./CVC_300/valid/labels/"
        output_dir_image = "./CVC_300/valid/images/"
        for i in range(48,len(image_list)):
          img_path = image_list[i]
          image_name = img_path.split('/')[-1].split(".")[0]
          img = Image.open(img_path)
          target_size = img.size
          csv_path = "./labels/"+ image_name +".csv"
          dfs= pd.read_csv(csv_path)
          lista = csv_txt(dfs, target_size)
          df = pd.DataFrame(lista)
          output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
          df.to_csv(output_file_path, sep='\t', index=False, header=False)
          shutil.move(img_path, output_dir_image)
            
    
    
################################################################################################################################
    elif  args.dataset == "PolypGen" :
    
        !mkdir -p "./polyp_gen/train/images"
        !mkdir -p "./polyp_gen/train/labels"
        !mkdir -p "./polyp_gen/valid/images"
        !mkdir -p "./polyp_gen/valid/labels"
        
        def csv_txt(dfs, target_size):
          x,y=target_size
          list=[]
          for w in range(len(dfs)):
            row = dfs.iloc[w]
            x_c = (row['x_max']+row['x_min'])/2/x
            y_c = (row['y_max']+row['y_min'])/2/y
            w = (row['x_max']-row['x_min'])/x
            h = (row['y_max']-row['y_min'])/y
            list.append([0,x_c,y_c,w,h])
          return list
        
        output_dir_label ="./polyp_gen/train/labels/"
        output_dir_image = "./polyp_gen/train/images/"
        output_dir_masks = "./polyp_gen/train/masks/"
        os.makedirs(output_dir_label, exist_ok=True)
        os.makedirs(output_dir_image, exist_ok=True)
        os.makedirs(output_dir_masks, exist_ok=True)
        image_list = glob.glob("./positive_cropped/*/images/*.jpg")
        for i in range(1780):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1].split(".")[0]
            sequence_name = img_path.split('/')[2]
            mask_add = "./"+ img_path.split('/')[1] + "/" + img_path.split('/')[2]+ "/masks/"+ img_path.split('/')[-1]
            img = Image.open(img_path)
            target_size=img.size
            mask_np = cv2.imread(mask_add, cv2.IMREAD_GRAYSCALE)
            mask_np = mask_np / 255
            seg_value = 1.
            segmentation = np.where(mask_np == seg_value)
            output_file_path = os.path.join(output_dir_label, sequence_name+image_name+ '.txt')
        
            bboxes = 0, 0, 0, 0
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
        
                bboxes = x_min, y_min, x_max, y_max
        
                num_masks = 1
                dfs = pd.DataFrame([[x_min, x_max, y_min, y_max]], columns=["x_min", "x_max", "y_min", "y_max"])
                lista = csv_txt(dfs, target_size)
                df = pd.DataFrame(lista)
                df.to_csv(output_file_path, sep='\t', index=False, header=False)
            else:
                open(output_file_path, 'a').close()
        
            image_destination = output_dir_image + sequence_name+image_name+ '.jpg'
            mask_destination = output_dir_masks + sequence_name+image_name+ '.jpg'
            shutil.move(img_path, image_destination)
            shutil.move(mask_add, mask_destination)
        
        
        
        valid_output_dir_label ="./polyp_gen/valid/labels/"
        valid_output_dir_image = "./polyp_gen/valid/images/"
        valid_output_dir_masks = "./polyp_gen/valid/masks/"
        os.makedirs(valid_output_dir_label, exist_ok=True)
        os.makedirs(valid_output_dir_image, exist_ok=True)
        os.makedirs(valid_output_dir_masks, exist_ok=True)
        for i in range(1780,len(image_list)):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1].split(".")[0]
            sequence_name = img_path.split('/')[2]
            mask_add = "./"+ img_path.split('/')[1] + "/" + img_path.split('/')[2]+ "/masks/"+ img_path.split('/')[-1]
            img = Image.open(img_path)
            target_size=img.size
            mask_np = cv2.imread(mask_add, cv2.IMREAD_GRAYSCALE)
            mask_np = mask_np / 255
            seg_value = 1.
            segmentation = np.where(mask_np == seg_value)
            output_file_path = os.path.join(output_dir_label, sequence_name+image_name+ '.txt')
        
            bboxes = 0, 0, 0, 0
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
        
                bboxes = x_min, y_min, x_max, y_max
        
                num_masks = 1
                dfs = pd.DataFrame([[x_min, x_max, y_min, y_max]], columns=["x_min", "x_max", "y_min", "y_max"])
                lista = csv_txt(dfs, target_size)
                df = pd.DataFrame(lista)
                df.to_csv(output_file_path, sep='\t', index=False, header=False)
            else:
                open(output_file_path, 'a').close()
        
        
            image_destination = output_dir_image + sequence_name+image_name+ '.jpg'
            mask_destination = output_dir_masks + sequence_name+image_name+ '.jpg'
            shutil.move(img_path, image_destination)
            shutil.move(mask_add, mask_destination)    
            
    
    
    
    
    
    
################################################################################################################################
    elif  args.dataset == "SUN-SEG" :

        all_images = glob.glob('./SUN-SEG/TrainDataset/Frame/*/*.jpg')
        output_dir_label = "./sun_seg/train/labels"
        output_dir_image = "./sun_seg/train/images"
        
        seg_value = 1.
        
        for i in range(len(all_images)):
            gt_path = all_images[i].split('/')
            gt_path[4]='GT'
            image_name = gt_path[-1].split('.')[0]
            gt_path[-1] = image_name + '.png'
            gt_path = '/'.join(gt_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)/255
            segmentation = np.where(mask_np == seg_value)
            heigth,width = mask_np.shape
            
            # Bounding Box
            bboxes = 0, 0, 0, 0
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
            
                bboxes = x_min, y_min, x_max, y_max
            
            X_C = (x_max+x_min)/2/width
            Y_C = (y_max+y_min)/2/heigth 
            W = (x_max - x_min)/width
            H = (y_max-y_min)/heigth 
            df = pd.DataFrame([[0, X_C, Y_C, W, H]])
            
            output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
            df.to_csv(output_file_path, sep='\t', index=False, header=False)
            shutil.copy(all_images[i], output_dir_image)
        
        
        
        
        all_images = glob.glob('./SUN-SEG/*/*/Frame/*/*.jpg')
        output_dir_label = "./sun_seg/valid/labels"
        output_dir_image = "./sun_seg/valid/images"
        
        seg_value = 1.
        
        for i in range(len(all_images)):
            gt_path = all_images[i].split('/')
            gt_path[5]='GT'
            image_name = gt_path[-1].split('.')[0]
            gt_path[-1] = image_name + '.png'
            gt_path = '/'.join(gt_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)/255
            segmentation = np.where(mask_np == seg_value)
            heigth,width = mask_np.shape
            
            # Bounding Box
            bboxes = 0, 0, 0, 0
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
            
                bboxes = x_min, y_min, x_max, y_max
            
            X_C = (x_max+x_min)/2/width
            Y_C = (y_max+y_min)/2/heigth 
            W = (x_max - x_min)/width
            H = (y_max-y_min)/heigth 
            df = pd.DataFrame([[0, X_C, Y_C, W, H]])
            
            output_file_path = os.path.join(output_dir_label, image_name+ '.txt')
            df.to_csv(output_file_path, sep='\t', index=False, header=False)
            shutil.copy(all_images[i], output_dir_image)



################################################################################################################################
    else:
        print("dataset not supported")


if __name__ == "__main__":
    main()