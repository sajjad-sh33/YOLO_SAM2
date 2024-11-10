from ultralytics import YOLO
from PIL import Image
import cv2
import glob
import torch
import numpy as np
import cv2
import os
import numpy as np
from sklearn.metrics import f1_score
from math import sqrt

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class StructureMeasure(object):
    def __init__(self):
        self.eps=np.finfo(np.double).eps

    def _Object(self,GT,pred):
        x=np.mean(pred[GT])
        sigma_x=np.std(pred[GT])
        score=2*x/(x*x+1+sigma_x+self.eps)
        return score

    def _S_object(self,GT,pred):
        #compute the similarity of the foreground
        pred_fg=pred.copy()
        pred_fg[~GT]=0
        O_FG=self._Object(GT,pred_fg)

        #compute the similarity of the background
        pred_bg=1-pred.copy()
        pred_bg[GT]=0
        O_BG=self._Object(~GT,pred_bg)

        #combine foreground and background
        u=np.mean(GT)
        Q=u*O_FG+(1-u)*O_BG
        return Q

    def _centroid(self,GT):
        rows,cols=GT.shape
        if np.sum(GT)==0:
            X=round(cols/2)
            Y=round(rows/2)
        else:
            total=np.sum(GT)
            i=range(cols)
            j=range(rows)
            X=int(round(np.sum(np.sum(GT,axis=0)*i)/total))+1
            Y=int(round(np.sum(np.sum(GT,axis=1)*j)/total))+1
        return (X,Y)

    def _divide_GT(self,GT,X,Y):
        rows,cols=GT.shape
        area=rows*cols
        LT=GT[0:Y,0:X]
        RT=GT[0:Y,X:cols]
        LB=GT[Y:rows,0:X]
        RB=GT[Y:rows,X:cols]

        w1=((X)*(Y))/area
        w2=((cols-X)*(Y))/area
        w3=((X)*(rows-Y))/area
        w4=1-w1-w2-w3
        return (LT,RT,LB,RB,w1,w2,w3,w4)

    def _divide_pred(self,pred,X,Y):
        rows, cols = pred.shape
        area = rows * cols
        LT = pred[0:Y, 0:X]
        RT = pred[0:Y, X:cols]
        LB = pred[Y:rows, 0:X]
        RB = pred[Y:rows, X:cols]
        return (LT, RT, LB, RB)

    def _ssim(self,GT,pred):
        rows,cols=GT.shape
        N=rows*cols
        x=np.mean(pred)
        y=np.mean(GT)
        sigma_x2 = np.sum((pred-x)**2)/(N-1+self.eps)
        sigma_y2 = np.sum((GT - y) ** 2) / (N - 1 + self.eps)
        sigma_xy=np.sum((pred-x)*(GT-y))/(N - 1 + self.eps)
        alpha=4*x*y*sigma_xy
        beta=(x**2+y**2)*(sigma_x2+sigma_y2)
        if alpha!=0:
            Q=alpha/(beta+np.finfo(np.double).eps)
        elif alpha==0 and beta==0:
            Q=1.0
        else:
            Q=0
        return Q

    def _S_region(self,GT,pred):
        X,Y=self._centroid(GT)
        GT_LT,GT_RT,GT_LB,GT_RB,w1,w2,w3,w4=self._divide_GT(GT,X,Y)

        Pred_LT,Pred_RT,Pred_LB,Pred_RB=self._divide_pred(pred,X,Y)
        Q1 = self._ssim(GT_LT,Pred_LT)
        Q2 = self._ssim(GT_RT, Pred_RT)
        Q3 = self._ssim(GT_LB, Pred_LB)
        Q4 = self._ssim(GT_RB, Pred_RB)
        Q=w1*Q1+w2*Q2+w3*Q3+w4*Q4
        return Q

    def _minmiax_norm(self,X,ymin=0,ymax=1):
        X = (ymax - ymin) * (X - np.min(X)) / (np.max(X) - np.min(X)) + ymin
        return X

    def _prepare_data(self,GT_path,pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.double)
        GT = np.array(Image.open(GT_path)).astype(np.bool)

        if len(pred.shape)!=2:
            pred=0.2989*pred[:,:,0]+0.5870*pred[:,:,1] + 0.1140*pred[:,:,2]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]
        #judge channel
        assert len(pred.shape)==2,"Pred should be one channel!"
        assert len(GT.shape)==2,"Ground Truth should be one channel!"
        #normalize
        if np.max(pred)==255:
            pred=(pred/255)
        pred=self._minmiax_norm(pred,0,1)
        return GT,pred

    def __call__(self,GT_path,pred_path):
        # GT,pred=self._prepare_data(GT_path,pred_path)
        GT = GT_path.astype(np.bool_)
        pred = pred_path.astype(np.double)
        meanGT=np.mean(GT)
        if meanGT==0:#ground truth is balck
            x=np.mean(pred)
            Q=1.0-x
        elif meanGT==1:#ground truth is white
            x=np.mean(pred)
            Q=x
        else:
            alpha=0.5
            Q=alpha*self._S_object(GT,pred)+(1-alpha)*self._S_region(GT,pred)
            if Q<0:
                Q=0
        return Q




class EnhancedAlignmentMeasure:
    def __init__(self):
        self.eps=np.finfo(np.double).eps

    def _prepare_data(self,GT_path,pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.bool)
        GT = np.array(Image.open(GT_path)).astype(np.bool)
        if len(pred.shape)!=2:
            pred=pred[:,:,0]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]

        #judge channel
        assert len(pred.shape)==2,"Pred should be one channel!"
        assert len(GT.shape)==2,"Ground Truth should be one channel!"
        return GT,pred

    def _EnhancedAlignmnetTerm(self,align_Matrix):
        enhanced=((align_Matrix+1)**2)/4
        return enhanced

    def _AlignmentTerm(self,dGT,dpred):
        mean_dpred=np.mean(dpred)
        mean_dGT=np.mean(dGT)
        align_dpred=dpred-mean_dpred
        align_dGT=dGT-mean_dGT
        align_matrix=2*(align_dGT*align_dpred)/(align_dGT**2+align_dpred**2+self.eps)
        return align_matrix

    def __call__(self,GT_path,pred_path):
        # GT,pred=self._prepare_data(GT_path,pred_path)
        GT = GT_path.astype(np.bool_)
        pred = pred_path.astype(np.double)
        dGT,dpred=GT.astype(np.float64),pred.astype(np.float64)
        if np.sum(GT)==0:#completely black
            enhanced_matrix=1-dpred
        elif np.sum(~GT)==0:
            enhanced_matrix=dpred
        else:
            align_matrix=self._AlignmentTerm(dGT,dpred)
            enhanced_matrix=self._EnhancedAlignmnetTerm(align_matrix)
        rows,cols= GT.shape
        score=np.sum(enhanced_matrix)/(rows*cols-1+self.eps)
        return score


def intersectionAndUnion(imPred, imLab, numClass):

	# imPred = imPred * (imLab>0)

	# Compute area intersection:
	intersection = imPred * (imPred==imLab)
	(area_intersection,_) = np.histogram(intersection, bins=numClass, range=(1, numClass))

	# Compute area union:
	(area_pred,_) = np.histogram(imPred, bins=numClass, range=(1, numClass))
	(area_lab,_) = np.histogram(imLab, bins=numClass, range=(1, numClass))
	area_union = area_pred + area_lab - area_intersection
	area_sum = area_pred + area_lab
    
	return (area_intersection, area_union, area_sum)

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


################################################################################################################################
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["Kvasir", "CVC-ColonDB", "CVC-ClinicDB", "ETIS", "CVC-300", "PolypGen", "SUN-SEG"], default="PolypGen",
                        help="Which downstream task.")
        parser.add_argument("--checkpoint_add", default="./runs/detect/yolov8l/weights/best.pt",
                        help="YOLO pre-trained model address.")
    args = parser.parse_args()


    yolo_model = YOLO(args.checkpoint_add) 
    

################################################################################################################################
if args.dataset == "Kvasir":
    numClass = 1
    image_list = image_list = glob.glob("./Kvasir/Kvasir/valid/images/*.jpg")
    arr=len(image_list)
    area_intersection = np.zeros((numClass, arr))
    area_union = np.zeros((numClass, arr))
    area_sum = np.zeros((numClass, arr))
    for i in range(arr):
        img_path = image_list[i]
        image_name = img_path.split('/')[-1]
        csv_path = "./Kvasir/Kvasir-SEG/Kvasir-SEG/masks/" + image_name  
        ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
        image = Image.open(image_list[i])
        results = yolo_model.predict([image], imgsz=640, conf=0.5)
        boxes = results[0].boxes
        if boxes.shape[0] == 1:
            boxes = boxes.cpu().numpy()
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)
            input_box = boxes.xyxy[0]
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            pred_mask = masks[0]
            (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
            print(i)
        if boxes.shape[0] > 1:
            boxes = boxes.cpu().numpy()
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)
            input_box = boxes.xyxy[0].reshape(-1,4)
            for z in range(boxes.shape[0]-1):
                input_box =np.concatenate((input_box, boxes.xyxy[z+1].reshape(-1,4)), axis=0)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            for k in range(len(masks)):
                if k == 0:
                    pred_mask = masks[k][0]
                else:
                    pred_mask = (pred_mask + masks[k][0]).clip(0,1)
            (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
            print(i)

    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
    Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
    print(f"Dataset = Kvasir2 | mIoU = {IoU} | mDice = {Dice}")
################################################################################################################################
    elif  args.dataset == "ColonDB" :


        numClass = 1
        image_list = image_list = glob.glob("./CVC_colondb/CVC_colondb/valid/images/*.png")
        arr=len(image_list)
        area_intersection = np.zeros((numClass, arr))
        area_union = np.zeros((numClass, arr))
        area_sum = np.zeros((numClass, arr))
        for i in range(arr):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1]
            csv_path = "./CVC_colondb/CVC-ColonDB/masks/" + image_name  
            ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
            image = Image.open(image_list[i])
            results = yolo_model.predict([image], imgsz=640, conf=0.5)
            boxes = results[0].boxes
            if boxes.shape[0] == 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
            if boxes.shape[0] > 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0].reshape(-1,4)
                for z in range(boxes.shape[0]-1):
                    input_box =np.concatenate((input_box, boxes.xyxy[z+1].reshape(-1,4)), axis=0)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0,1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        print(f"Dataset = CVC_colondb2 | mIoU = {IoU} | mDice = {Dice}")

################################################################################################################################
    elif  args.dataset == "CVC-ClinicDB" :

        numClass = 1
        image_list = glob.glob("./CVC_clinicdb/CVC_clinicdb/valid/images/*.png")
        arr = len(image_list)
        area_intersection = np.zeros((numClass, arr))
        area_union = np.zeros((numClass, arr))
        area_sum = np.zeros((numClass, arr))
        
        for i in range(arr):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1]
            csv_path = "./CVC_clinicdb/PNG/Ground Truth/" + image_name  
            ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
            image = Image.open(image_list[i])
            results = yolo_model.predict([image], imgsz=640, conf=0.5)
            boxes = results[0].boxes
            if boxes.shape[0] == 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
            if boxes.shape[0] > 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0].reshape(-1, 4)
                for z in range(boxes.shape[0] - 1):
                    input_box = np.concatenate((input_box, boxes.xyxy[z + 1].reshape(-1, 4)), axis=0)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0, 1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title(f'Main Image: {image_name}')
            plt.imshow(image)
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth Mask')
            plt.imshow(ground_truth, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title('Predicted Mask')
            plt.imshow(pred_mask, cmap='gray')
            plt.show()
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        print(f"Dataset = CVC_clinicdb2 | mIoU = {IoU} | mDice = {Dice}")

        
################################################################################################################################
    elif  args.dataset == "ETIS" :
   
    
        numClass = 1
        image_list = image_list = glob.glob("./ETIS/ETIS/valid/images/*.png")
        arr=len(image_list)
        area_intersection = np.zeros((numClass, arr))
        area_union = np.zeros((numClass, arr))
        area_sum = np.zeros((numClass, arr))
        for i in range(arr):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1]
            csv_path = "./ETIS/masks/" + image_name  
            ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
            image = Image.open(image_list[i])
            results = yolo_model.predict([image], imgsz=640, conf=0.5)
            boxes = results[0].boxes
            if boxes.shape[0] == 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
            if boxes.shape[0] > 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0].reshape(-1,4)
                for z in range(boxes.shape[0]-1):
                    input_box =np.concatenate((input_box, boxes.xyxy[z+1].reshape(-1,4)), axis=0)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0,1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        print(f"Dataset = ETIS2 | mIoU = {IoU} | mDice = {Dice}")

    

################################################################################################################################    
    elif  args.dataset == "CVC-300" :

        numClass = 1
        image_list = image_list = glob.glob("./CVC_300/CVC-300/valid/images/*.png")
        arr=len(image_list)
        area_intersection = np.zeros((numClass, arr))
        area_union = np.zeros((numClass, arr))
        area_sum = np.zeros((numClass, arr))
        for i in range(arr):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1]
            csv_path = "./CVC_300/CVC-300/masks/" + image_name  
            ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
            image = Image.open(image_list[i])
            results = yolo_model.predict([image], imgsz=640, conf=0.5)
            boxes = results[0].boxes
            if boxes.shape[0] == 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
            if boxes.shape[0] > 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0].reshape(-1,4)
                for z in range(boxes.shape[0]-1):
                    input_box =np.concatenate((input_box, boxes.xyxy[z+1].reshape(-1,4)), axis=0)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0,1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                print(i)
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        print(f"Dataset = CVC_3002 | mIoU = {IoU} | mDice = {Dice}")
    
################################################################################################################################
    elif  args.dataset == "PolypGen" :
    
        numClass = 1
        image_list = glob.glob("./polyp_gen/valid/images/*.jpg")
        arr=len(image_list)
        area_intersection = np.zeros((numClass, arr))
        area_union = np.zeros((numClass, arr))
        area_sum = np.zeros((numClass, arr))
        TP_FN = 0
        TP_FP = 0
        
        for i in range(arr):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1]
            csv_path = "./polyp_gen/valid/masks/" + image_name
            ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
            image = Image.open(image_list[i])
            results = yolo_model.predict([image], imgsz=640, conf=0.5)
            boxes = results[0].boxes
            if boxes.shape[0] < 1:
                pred_mask = np.zeros(ground_truth.shape)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                TP_FN = TP_FN + np.sum(ground_truth)
                TP_FP = TP_FP + np.sum(pred_mask)
                print(i)
            if boxes.shape[0] == 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                TP_FN = TP_FN + np.sum(ground_truth)
                TP_FP = TP_FP + np.sum(pred_mask)
                print(i)
            if boxes.shape[0] > 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0].reshape(-1,4)
                for z in range(boxes.shape[0]-1):
                    input_box =np.concatenate((input_box, boxes.xyxy[z+1].reshape(-1,4)), axis=0)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0,1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                TP_FN = TP_FN + np.sum(ground_truth)
                TP_FP = TP_FP + np.sum(pred_mask)
                print(i)
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        
        recal = 1.0 * np.sum(area_intersection, axis=1) / TP_FN
        precision = 1.0 * np.sum(area_intersection, axis=1) / TP_FP
        beta2 = 4
        F_score = (1+beta2)*precision*recal / ((beta2 *precision)+recal)
        
        
        print(f"Dataset = polypgen_n2 | mIoU = {IoU} | Dice ={Dice}| recal = {recal} |precision={precision} |F_score={F_score} ")
            
            
    
################################################################################################################################
    elif  args.dataset == "SUN-SEG" :

        numClass = 1
        image_list  = glob.glob("./data/SUN-SEG/TestEasyDataset/Seen/Frame/*/*.jpg")
        arr=len(image_list)
        area_intersection = np.zeros((numClass, arr))
        area_union = np.zeros((numClass, arr))
        area_sum = np.zeros((numClass, arr))
        TP_FN = 0
        TP_FP = 0
        S_measures = 0
        E_measures = 0
        counter = 0
        S_measure = StructureMeasure()
        E_measure = EnhancedAlignmentMeasure()
        
        for i in range(arr):
            img_path = image_list[i]
            image_name = img_path.split('/')[-1]
            csv_path =  "./data/SUN-SEG/TestEasyDataset/Seen/GT/" + img_path.split('/')[-2] + '/' + img_path.split('/')[-1].split('.')[0] + '.png'
            ground_truth = cv2.imread(csv_path, cv2.IMREAD_GRAYSCALE) / 255
            image = Image.open(image_list[i])
            results = yolo_model.predict([image], imgsz=640, conf=0.5)
            boxes = results[0].boxes
            if boxes.shape[0] == 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                TP_FN = TP_FN + np.sum(ground_truth)
                TP_FP = TP_FP + np.sum(pred_mask)
                S_measures = S_measures + S_measure(ground_truth,pred_mask)
                E_measures = E_measures + E_measure(ground_truth,pred_mask)
                counter +=1
                print(i)
            if boxes.shape[0] > 1:
                boxes = boxes.cpu().numpy()
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                input_box = boxes.xyxy[0].reshape(-1,4)
                for z in range(boxes.shape[0]-1):
                    input_box =np.concatenate((input_box, boxes.xyxy[z+1].reshape(-1,4)), axis=0)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0,1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask, ground_truth, numClass)
                TP_FN = TP_FN + np.sum(ground_truth)
                TP_FP = TP_FP + np.sum(pred_mask)
                S_measures = S_measures + S_measure(ground_truth,pred_mask)
                E_measures = E_measures + E_measure(ground_truth,pred_mask)
                counter +=1
                print(i)
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        
        recal = 1.0 * np.sum(area_intersection, axis=1) / TP_FN
        precision = 1.0 * np.sum(area_intersection, axis=1) / TP_FP
        beta2 = 0.3
        F_score = (1+beta2)*precision*recal / ((beta2 *precision)+recal)
        
        S_measures = S_measures / counter
        E_measures = E_measures / counter
        
        print(f"Dataset = Kvasir2 | mIoU = {IoU} | mDice = {Dice} | S_measure = {S_measures} | E_measure = {E_measures} | recal = {recal} |precision={precision} |F_score={F_score} ")


################################################################################################################################
    else:
        print("dataset not supported")


if __name__ == "__main__":
    main()
