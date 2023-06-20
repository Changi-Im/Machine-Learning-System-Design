import numpy as np
import pandas as pd

if __name__ == "__main__":
    W = np.random.random((4,3))
    X = np.random.random((4,4))
    B = np.random.random((3,4))
    b = 1
    
    Y0 = np.linalg.inv(np.dot(X.T, X))
    Y1 = np.dot(W.T,X)
    Y4 = np.where(Y1 > 1, 1, 0)
    
    data = {'year':[2016, 2017, 2018],
            'car': ["그랜저", "그랜저", "소나타"],
            'name': ["홍길동", "고길동", "김둘리"],
            'number': ['123하4567', '123허4567', '123호4567']}
    
    df = pd.DataFrame(data)
    
    d = {'year':2017,
         'car': "테슬라",
         'name': "일론",
         'number': '987하6543'}
    
    df.loc()[3] = d
    print(df)
    print(df[['year', 'car', 'number']])
    print(df[df['year']<2018])
    
    
    data1 = "2014년.csv"
    data2 = "./2015년.csv"
    data3 = "./2016년.csv"
    
    df1 = pd.read_csv(data1, index_col = 0, encoding='cp949')
    df2 = pd.read_csv(data2, index_col = 0, encoding='cp949')
    df3 = pd.read_csv(data3, index_col = 0, encoding='cp949')
    
    df = pd.concat([df1, df2, df3])
    
    index = []
    
    for i in range(0,len(df.index)):
        yr, mon = df.index[i].replace("년"," ").strip("월").split()
        
        index.append((yr, mon))
    
    df.index = pd.MultiIndex.from_tuples(index, names = ["년도", "월"])
    
    year = df.groupby("년도").mean()
    month = df.groupby("월").mean()

    print(year["사망(명)"])
    print(month["사망(명)"])    
    
    d = df.groupby("년도").sum()
    rate = d.loc["2016"]["사망(명)"] / d.loc["2016"]["사고(건)"] * 100
    print(rate)
    
    
    