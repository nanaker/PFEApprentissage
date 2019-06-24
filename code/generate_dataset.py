import pandas as pd
df = pd.read_csv('../dataset/LIC/LIC_.csv')
liste=["True","False"]
elem=0;
lignes=[]
for bool1 in liste:
    for bool2 in liste:
        for bool3 in liste:
            for bool4 in liste:
              for bool5 in liste :
                for bool6 in liste:
                    for bool7 in liste:
                        ligne=bool1+","+bool2+","+bool3+","+bool4+","+bool5+","+bool6+","+bool7+",false"
                        if ligne!=("False,False,False,False,False,False,False,false"):
                            df = df.append(pd.Series([bool1, bool2, bool3, bool4,bool5,bool6,bool7,"false"],
                                                             index=df.columns),
                                                   ignore_index=True)
                            lignes.append(ligne)


print(lignes.__len__())
for ligne in lignes:
    print(ligne)
print(df['is_code_smell'].describe())


df.to_csv('../dataset/LIC/LIC_new.csv', index=False)





