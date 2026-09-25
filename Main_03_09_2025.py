import pandas as pd
from numpy import matlib
from FSA import FSA
from Global_Vars import Global_Vars
from Image_Results import *
from LEA import LEA
from MobileFaceNet import Model_MobileFaceNet
from Model_CNN import Model_CNN
from Model_CNN_LSTM import Model_CNN_LSTM
from Model_ConvNet import Model_ConvNet
from Model_Ensemble import Model_Ensemble
from Model_ResNet import Model_ResNet
from Model_ResNet50 import Model_ResNet50
from Model_Vgg16 import Model_VGG16
from OOA import OOA
from Objfun import objfun_2, objfun_1
from PCOA import PCOA
from Plot_Results import *
from Proposed import Proposed

No_of_Dataset = 3

# Read Dataset 1
an = 0
if an == 1:
    Image = []
    Target = []
    Path = './Dataset_1/test'
    out_dir = os.listdir(Path)
    for i in range(len(out_dir)):
        folder = Path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            FileName = folder + '/' + in_dir[j]
            img = cv.imread(FileName)
            Resize_Img = cv.resize(img, (512, 512))
            Image.append(Resize_Img)
            Target.append(i)

    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_1.npy', index)
    np.save('Image_1.npy', Shuffled_Datas)
    np.save('Target_1.npy', Shuffled_Target)

# Read Dataset 2
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Dataset_2/images'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            FileName = folder + '/' + in_dir[j]
            split_data = FileName.split('/')
            if split_data[4] == 'Anger.jpg':
                Target.append(0)
            elif split_data[4] == 'Contempt.jpg':
                Target.append(1)
            elif split_data[4] == 'Disgust.jpg':
                Target.append(2)
            elif split_data[4] == 'Fear.jpg':
                Target.append(3)
            elif split_data[4] == 'Happy.jpg':
                Target.append(4)
            elif split_data[4] == 'Neutral.jpg':
                Target.append(5)
            elif split_data[4] == 'Sad.jpg':
                Target.append(6)
            else:
                Target.append(7)
            img = cv.imread(FileName)
            Resize_Img = cv.resize(img, (512, 512))
            Image.append(Resize_Img)

    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_2.npy', index)
    np.save('Image_2.npy', Shuffled_Datas)
    np.save('Target_2.npy', Shuffled_Target)

# Read Dataset 3
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Dataset_3/images/images/train'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            FileName = folder + '/' + in_dir[j]
            img = cv.imread(FileName)
            Resize_Img = cv.resize(img, (512, 512))
            Image.append(Resize_Img)
            Target.append(i)

    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_3.npy', index)
    np.save('Image_3.npy', Shuffled_Datas)
    np.save('Target_3.npy', Shuffled_Target)

# Image Preprocessing
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Pre_Image = []
        Cla_Img = []
        for i in range(len(Image)):
            print(i)
            Img = Image[i]
            image_bw = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=5)
            clahe_img = np.clip(clahe.apply(image_bw), 0, 255).astype(np.uint8)
            mean = np.mean(clahe_img)
            std_dev = np.std(clahe_img)
            normalized_image = ((clahe_img - mean) / std_dev).astype(np.uint8)
            Cla_Img.append(clahe_img)
            Pre_Image.append(normalized_image)
        np.save('Pre_Image_' + str(n + 1) + '.npy', np.asarray(Pre_Image))

# Optimization for Recognition
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Pre_Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron, Learning Rate, Activation Function
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = objfun_1
        Max_iter = 50

        print("PCOA...")
        [bestfit1, fitness1, bestsol1, time1] = PCOA(initsol, fname, xmin, xmax, Max_iter)

        print("LEA...")
        [bestfit2, fitness2, bestsol2, time2] = LEA(initsol, fname, xmin, xmax, Max_iter)

        print("FSA...")
        [bestfit3, fitness3, bestsol3, time3] = FSA(initsol, fname, xmin, xmax, Max_iter)

        print("OOA...")
        [bestfit4, fitness4, bestsol4, time4] = OOA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('Best_Sol_' + str(n + 1) + '.npy', BestSol)

# Recognition
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Image = np.load('Pre_Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('Best_Sol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Epochs = [20, 40, 60, 80, 100]
        for act in range(len(Epochs)):
            learnperc = round(Image.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Image[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Image[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 11))
            for j in range(BestSol.shape[0]):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_MobileFaceNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epochs[act], sol)  # MobileFaceNet With optimization
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                              Epochs[act])
            Eval[6, :], pred2 = Model_ResNet(Train_Data, Train_Target, Test_Data,
                                             Test_Target, Epochs[act])
            Eval[7, :], pred3 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epochs[act])
            Eval[8, :], pred4 = Model_MobileFaceNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epochs[act])  # MobileFaceNet Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Evaluate_Seg_all.npy', Eval_all)

# Optimization for Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Pre_Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron, Epoch, Learning Rate
        xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
        xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = objfun_2
        Max_iter = 50

        print("PCOA...")
        [bestfit1, fitness1, bestsol1, time1] = PCOA(initsol, fname, xmin, xmax, Max_iter)

        print("LEA...")
        [bestfit2, fitness2, bestsol2, time2] = LEA(initsol, fname, xmin, xmax, Max_iter)

        print("FSA...")
        [bestfit3, fitness3, bestsol3, time3] = FSA(initsol, fname, xmin, xmax, Max_iter)

        print("OOA...")
        [bestfit4, fitness4, bestsol4, time4] = OOA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feat = np.load('Pre_Image_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        EVAL = []
        Activation_Function = ['Linear', 'ReLu', 'Softmax']
        for act in range(len(Activation_Function)):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target,
                                              Activation_Function[act], sol)  # With optimization
            Eval[5, :], pred1 = Model_ConvNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                              Activation_Function[act]) 
            Eval[6, :], pred2 = Model_ResNet50(Train_Data, Train_Target, Test_Data,
                                             Test_Target, Activation_Function[act])
            Eval[7, :], pred3 = Model_CNN_LSTM(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Activation_Function[act])
            Eval[8, :], pred4 = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target,
                                              Activation_Function[act])  # Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        np.save('Eval_all.npy', EVAL)  # Save Eval all

plotConvResults()
plot_Alg_Results()
Plot_Mod_Results()
Plot_ROC_Curve()
Table()
Proposed_Plots_Results()
Reg_Plot_Results()
Reg_Table()
Sample_Images()
