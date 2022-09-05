def PCA_Method(NeedPercent,Dimension):
    #------------------------------------------------------
    #             Read  And Show  DataSet                  
    #------------------------------------------------------
    import pandas as pd
    import numpy as np
    Data_Set= pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\F_Data_Set.txt')
    Data_Set.head()
    #------------------------------------------------------
    #                  END
    #------------------------------------------------------



    #------------------------------------------------------
    #           z_score   Normalization                  
    #------------------------------------------------------
    Data_Set.describe().round()
    z_score = (Data_Set - Data_Set.mean())/Data_Set.std()
    Data_Scaled=z_score
    Data_Scaled.head()
    #------------------------------------------------------
    #                  END
    #------------------------------------------------------





    #------------------------------------------------------
    #             Calculating  AND Show  covariance               
    #------------------------------------------------------
    import numpy as np
    import seaborn as sn
    import matplotlib.pyplot as plt
    Cov_Data_Scaled=np.cov(Data_Scaled.T,bias=True)
    sn.heatmap(Cov_Data_Scaled, annot=True, fmt='g')
    plt.show()
    #------------------------------------------------------
    #                  END
    #------------------------------------------------------





    #------------------------------------------------------
    #     Calculating  AND Show eign_vals, eign_vecs              
    #------------------------------------------------------
    import matplotlib.pyplot as plt
    eign_vals, eign_vecs = np.linalg.eig(Cov_Data_Scaled)
    print('\nEignvalues \n%s' % eign_vals,'\n\n eign_vecs\n')
    print(eign_vecs)
    plt.bar(range(0,4), eign_vals, alpha = 0.5, align= 'center', label= 'indivisual eign_vals')
    plt.show()

   #------------------------------------------------------
   #                 END         
   #------------------------------------------------------




   #------------------------------------------------------
   #     Calculating Steps For My Percent            
   #------------------------------------------------------
    Percentlist=[]
    eign_valsist=list(eign_vals)
    MyPercent=0
    Sum_eignvals=sum(eign_valsist)
    for i in range(0,len(eign_valsist)):
        Percent1=eign_valsist[i]/Sum_eignvals
        Percent2=round(Percent1,3)
        print('\nPercent  :%s' %Percent2,'\n')
        Percentlist.append(Percent2)
        MyPercent=MyPercent+Percentlist[i]
        if  MyPercent>=NeedPercent:
            MySteps=i+1
            break
    print('\nNumber OF My PCA  :%s' %MySteps,'\n')
    print('\nPercent_list OF PCA :%s' %Percentlist,'\n')
    print('\nSum OF Percent_list :%s' %sum(Percentlist),'\n')
    print('------------------------------------------\n')    
    #------------------------------------------------------
    #                 END         
    #------------------------------------------------------


    #------------------------------------------------------
    #     Calculating PCA Method              
    #------------------------------------------------------
    PCA=0
    PCAList=[]
    Steps=0
    #-----------------------
    def column(eign_vecs, i):
        return [row[i] for row in eign_vecs]
    #-----------------------
    for i in range(0,MySteps):
        Feature=Data_Scaled.dot(column(eign_vecs,i))
        PCA1=np.var(Feature)
        PCA2=round(PCA1,3)
        PCAList.append(PCA2)
        PCA=PCA+PCA2
        Steps=i+1
        print('------------------------------------------\n')
        print('\nNumber OF PCA :%s' %Steps,'\n')
        print('\nPCA_Vector \n%s'     %Feature,'\n')
        print('\nPCA_Var \n%s'        %PCA2,'\n')
        print('\nPCA_Var_List \n%s'   %PCAList,'\n')
        print('------------------------------------------\n')

    




       #------------------------------------------------------
       #                 END         
       #------------------------------------------------------        


    
       #------------------------------------------------------
       #     Show PCA Method              
       #------------------------------------------------------    
    cum_var_PCA = np.cumsum(Percentlist)
    import matplotlib.pyplot as plt
    plt.bar(range(0, MySteps), Percentlist, alpha = 0.5, align= 'center', label= 'indivisual var_exp')
    plt.step(range(0,MySteps), cum_var_PCA, where= 'mid', label= 'cumulative var_exp')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc = 'best')
    plt.show()
   #------------------------------------------------------
   #                 END         
   #------------------------------------------------------


    if Dimension==2:
        
        #------------------------------------------------------
        #      Data Visualization   In 2D                    
        #------------------------------------------------------    
        x =Data_Scaled.dot(column(eign_vecs,0))
        y =Data_Scaled.dot(column(eign_vecs,1))
        plt.plot(x, y, label='data points', color='green', linestyle='dashed', linewidth = 2, 
             marker='*', markerfacecolor='blue', markersize=15)
        plt.ylim(-3,3) 
        plt.xlim(-3,3) 
        plt.xlabel('Feature_1') 
        plt.ylabel('Feature_2') 
        plt.title('My PCA') 
        plt.legend() 
        plt.show() 
        #------------------------------------------------------
        #                 END         
        #------------------------------------------------------
    if Dimension==3:
        
        #------------------------------------------------------
        #      Data Visualization   In 3D                    
        #------------------------------------------------------    
        x = np.linspace(-3, 2, 2)
        y = np.linspace(-2, 2, 2)
        z = np.linspace(-2, 2, 2)
        xdata = Data_Scaled.dot(column(eign_vecs,0))
        ydata = Data_Scaled.dot(column(eign_vecs,1))
        zdata = Data_Scaled.dot(column(eign_vecs,2))
        print(xdata)
        print(ydata)
        print(zdata)
        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues');
        #------------------------------------------------------
        #                 END         
        #------------------------------------------------------










