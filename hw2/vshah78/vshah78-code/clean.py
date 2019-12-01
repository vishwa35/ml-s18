import pandas as pd

def load_data(dir):
    df = pd.read_csv(dir, index_col=0)
    df = preproccess(df)
    # counter = Counter(np.ravel(labels))
    # print (Counter)
    # train, test, labels, answers = train_test_split(df, labels, stratify=labels.intent, test_size=0.3)
    return df

def preproccess(df):
    df = df.loc[df['year'] == 2014]
    df = df.loc[df['month'] == 1]
    df = df.drop(['month', 'year'], axis=1)
    df['sex'] = (df['sex'] == 'M').astype(int)
    df = pd.get_dummies(df, columns=["race"])
    df = pd.get_dummies(df, columns=["place"])
    df = pd.get_dummies(df, columns=["education"])
    df.loc[df['hispanic'] <= 199, 'hispanic'] = "not-hispanic"
    df.loc[df['hispanic'] <= 209, 'hispanic'] = "spaniard"
    df.loc[df['hispanic'] <= 219, 'hispanic'] = "mexican"
    df.loc[df['hispanic'] <= 229, 'hispanic'] = "central-american"
    df.loc[df['hispanic'] <= 249, 'hispanic'] = "south-american"
    df.loc[df['hispanic'] <= 259, 'hispanic'] = "latin-american"
    df.loc[df['hispanic'] <= 269, 'hispanic'] = "puerto_rican"
    df.loc[df['hispanic'] <= 274, 'hispanic'] = "cuban"
    df.loc[df['hispanic'] <= 279, 'hispanic'] = "dominican"
    df.loc[df['hispanic'] <= 299, 'hispanic'] = "other-spanish-hispanic"
    df.loc[df['hispanic'] <= 999, 'hispanic'] = "unknown"
    df = pd.get_dummies(df, columns=["hispanic"])
    df = pd.get_dummies(df, columns=["intent"])
    df = df.dropna(thresh=len(df.columns))
    # labels = df.filter(['intent'], axis=1)
    # df = df.drop("intent", axis=1)

    return df


df = load_data("ABAGAIL/guns.csv")

df.to_csv("ABAGAIL/gundeaths.txt")