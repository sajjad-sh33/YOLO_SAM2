from ultralytics import YOLO
from PIL import Image
import cv2
import glob
import torch
import numpy as np
import cv2
import os
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils.validation import EnhancedAlignmentMeasure, StructureMeasure

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
    if args.dataset == "Kvasir" and :


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
                input_boxes = torch.tensor(input_box, device=predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                for k in range(len(masks)):
                    if k == 0:
                        pred_mask = masks[k][0]
                    else:
                        pred_mask = (pred_mask + masks[k][0]).clip(0,1)
                (area_intersection[:, i], area_union[:, i], area_sum[:, i]) = intersectionAndUnion(pred_mask.numpy(), ground_truth, numClass)
                print(i)
        
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
        Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
        print(f"Dataset = Kvasir1 | mIoU = {IoU} | mDice = {Dice}")

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