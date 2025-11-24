import numpy as np
import pandas as pd



# Read Dataset for crop yeild pridiction
an = 1
if an == 1:
    dataset = './Data/yield.csv'
    df = pd.read_csv(dataset)
    Target = df['Item'].values
    df.drop(['Item'], inplace=True, axis=1)
    data = df.values
    for i in range(data.shape[1]):
        if data[:, i][0] == str(data[:, i][0]):
            uniq = np.unique(data[:, i])
            Datas = data[:, i]
            Uni_data = np.zeros((Datas.shape[0]))  # create within rage zero values

            for uni in range(len(uniq)):
                index = np.where(Datas == uniq[uni])
                Uni_data[index[0]] = uni + 1
            data[:, i] = Uni_data

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    targets = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        targets[index[0], uni] = 1
    index = np.arange(len(data))
    np.random.shuffle(index)
    Org_data = np.asarray(data)
    Shuffled_Datas = Org_data[index]
    Shuffled_Target = targets[index]
    np.save('Yield_Data.npy', Shuffled_Datas)
    np.save('Yield_Target.npy', Shuffled_Target)