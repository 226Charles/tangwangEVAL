import glob
import json
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score

EPS = 1e-6

# 读取企业标注结果的Excel文件
excel_path = '/workspace/enterprise/annotation_result.xlsx'

json_path = './workspace/**/classification.json'
json_files = glob.glob(json_path,recursive=True)

def inputDel(fileName):

    classificationDatas = []

    for filename in fileName:
        with open(filename, 'r', encoding='utf-8') as file:
            classificationDatas.append(json.load(file))

    #diabetic retinopathy
    diabeRets = []
    for i, classificationData in enumerate(classificationDatas):
        temp1 = dataDel(classificationData)
        diabeRets.append(temp1)

    return diabeRets

def dataDel(classificationData):

    result_dict = {}

    image_class_key = list(classificationData["imageClass"].keys())[0]
    uid = classificationData["imageClass"][image_class_key][0]["uid"]
    type = classificationData["imageClass"][image_class_key][0]["labelContentObject"]["F01"][0]["name"]

    result_dict[uid] = type
    
    return result_dict

def TFNP(gtDiabeRets,aiDiabeRets):

    sorted_gtDiabeRets = sorted(gtDiabeRets, key=lambda x: list(x.keys())[0])
    sorted_aiDiabeRets = sorted(aiDiabeRets, key=lambda x: list(x.keys())[0])

    mergeDiabeRets = []
    for i, item in enumerate(sorted_gtDiabeRets):
        uid = list(item.keys())[0]
        type = "hit"
        if sorted_gtDiabeRets[i][uid] == "转诊":
            if sorted_aiDiabeRets[i][uid] == "转诊":
                type = "TP"
            else:
                type = "FN"
        else:
            if sorted_aiDiabeRets[i][uid] == "转诊":
                type = "FP"
            else:
                type = "TN"
        temp = {}
        temp[uid] = type
        mergeDiabeRets.append(temp)

    return mergeDiabeRets

def claim_eval(mergeDiabeRets,listGt,listAi,y_true,y_scores):
    #从merged_result_dict里抓TP TN FP FN
    hit_mat = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for item in mergeDiabeRets:
        uid = list(item.keys())[0]
        hit_mat[item[uid]] += 1
    TP, TN, FP, FN = [hit_mat[key] for key in ["TP", "TN", "FP", "FN"]]
    
    result = {}
    result["sen"] = (TP) / (TP + FN + EPS)
    result["spe"] = (TN) / (TN + FP + EPS)
    result["acc"] = (TP + TN) / (TP + FP + TN + FN + EPS)
    result["kappa"] = cohen_kappa_score(listGt, listAi)
    result["auc"] = ROC_auc(y_true, y_scores)

    return result

def diabeRests_To_List(diabeRets):

    listAns = []
    for i, item in enumerate(diabeRets):
        uid = list(item.keys())[0]
        listAns.append(diabeRets[i][uid])

    return listAns

def ROC_auc(y_true, y_scores):
    # 计算ROC曲线的假阳性率（FPR）和真阳性率（TPR）
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # 计算AUC值
    auc = roc_auc_score(y_true, y_scores)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('./src/roc_curve.png', dpi=300)

    return auc

def reName(listGt):

    y_true = []
    for item in listGt:
        if item == "转诊":
            y_true.append(1)
        else:
            y_true.append(0)

    return y_true

def write_Excel(output_dir,data,title):

    ans = [data]
    sheet_name = "评价结果"

    df = pd.DataFrame(ans, columns=title)
    excel_writer = pd.ExcelWriter(output_dir + '/' + 'result.xlsx', engine='openpyxl')
    df.to_excel(excel_writer, index=False, sheet_name=sheet_name)
    excel_writer._save()

def mainControl():

    #企业分类预测概率 目前没有
    y_scores = [0.93928998708725,0.93928998708725,0.93928998708725,0.84355435135334]
    output_dir = "./workspace/classification_result"

    gtDiabeRets = inputDel(json_files)
    aiDiabeRets = gtDiabeRets #暂时
    mergeDiabeRets = TFNP(gtDiabeRets,aiDiabeRets)
    listGt = diabeRests_To_List(gtDiabeRets)
    listAi = diabeRests_To_List(aiDiabeRets)
    y_true = reName(listGt)
    result = claim_eval(mergeDiabeRets,listGt,listAi,y_true,y_scores)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_excel_title = ["sen","spe","acc","kappa","auc"]
    write_Excel(output_dir,result,output_excel_title)

if __name__ == '__main__':

    mainControl()

