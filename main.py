from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from matplotlib import cm
import gdal
from sklearn.cluster import KMeans
import sys
import os

objective = []
class KMEANS:

    def __init__(self):
        print('K means initiated')
        self.dataset = None
        self.band = None
        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.ret = None
        self.label = None
        self.center = None
        self.result = None
        self.reshaped = None
        self.res = None
        self.k = None
        self.bandOneMin = None
        self.bandTwoMin = None
        self.bandThreeMin = None
        self.bandOneMax = None
        self.bandTwoMax = None
        self.bandThreeMax = None
        self.clusters = []
        self.name =sys.argv[2].split('.')[0]
        self.path = 'D:/SIH 2019/isro/'+ self.name +'/'
        self.allcluster = None
        self.colorval = 0
        self.color = [[255,255,0],[255,0,0],[0,255,0],[0,0,255],[0,255,255],[255,0,255],[0,0,0],[255,255,255],[0,0,128],[192,192,192],[192,19,12],[242,192,22],[42,12,192],[82,82,12],[252,56,192],[182,12,112],[76,192,15]]
        self.eff = 0

        try:
            os.makedirs(self.path)
        except FileExistsError:
            # directory already exists
            pass


    def loadImage(self,imagename):
        print('Loading Image File -',imagename)
        self.dataset = gdal.Open(imagename)
        self.band = self.dataset.GetRasterBand(1)
        print(self.band.ComputeRasterMinMax(True))
        self.img1 = self.band.ReadAsArray()
        self.bandOneMin = np.percentile(self.img1,2)
        self.bandOneMax = np.percentile(self.img1,98)
        self.img1 = np.float32(self.img1)

        self.dataset = gdal.Open(imagename)
        self.band = self.dataset.GetRasterBand(2)
        self.img2 = self.band.ReadAsArray()
        self.bandTwoMin = np.percentile(self.img2,2)
        self.bandTwoMax = np.percentile(self.img2,98)
        self.img2 = np.float32(self.img2)

        self.dataset = gdal.Open(imagename)
        self.band = self.dataset.GetRasterBand(3)
        self.img3 = self.band.ReadAsArray()
        self.bandThreeMin = np.percentile(self.img3,2)
        self.bandThreeMax = np.percentile(self.img3,98)
        self.img3 = np.float32(self.img3)





    def saveImage(self,filename,image):
        print('Saving image as ',filename)
        cv2.imwrite(filename,image)



    def ObjectiveFunction(self,k,hstack,label,center):
        summ = np.zeros_like(hstack[0])
        for c in range(k):
            for i in range(len(label)):
                if(label[i]==c):
                    diff = center[c] - hstack[i]
                    diff = diff * diff
                    summ = summ + diff
        print('k - ',str(k),'summ - ',summ)
        # print(sum(summ))
        return sum(summ)




    def overlayRed(self,imagename,value=200):
        print('overlaying red')
        temp = self.result.reshape((-1, 1))
        X = np.zeros_like(self.res)
        for i in range(len(X)):
            if(temp[i]>=value):
                X[i] = self.reshaped[i]
        # for i in range(len(X)):
        #     print(X[i])
        X = X.reshape((self.img1.shape))
        self.saveImage(imagename,X)


    def arrayToText(self,filename,arr):
        file = open(filename,'w')
        for i in range(len(arr)):
            file.write(str(arr[i])+'\n')
        file.close()

        # file = open('writer.txt','w')
        # for i in range(len(pic)):
        #     for j in range(len(pic[0])):
        #         file.write(str(pic[i][j])+'\t')
        #     file.write('\n')
        # file.close()


    def getResultImage(self,filename,image):
        r = image[:, 0]
        g = image[:, 1]
        b = image[:, 2]


    def details(self):
        print('img1')
        print(self.img1)
        print('img2')
        print(self.img2)
        print('img3')
        print(self.img3)
        print('img size')
        print(self.img1.shape)
        print('ret')
        print(self.ret)
        print('label')
        print(self.label)
        print('center')
        print(self.center)
        print('result')
        print(self.result )
        print('reshaped')
        print(self.reshaped)
        print('Res')
        print(self.res    )
        print('K - value')
        print(self.k   )

    def histogram(self,filename,arr):
        print('Data Histogram - '+filename)
        plt.hist(arr)
        plt.ylabel('Density')
        plt.savefig(filename)

    def seperate(self):
        # print('result')
        # print(self.res.shape)
        # print('center')
        # print(self.center)
        file = open(os.path.join(self.path, 'center.txt'),'w')
        for i in range(len(self.center)):
            file.write(str(self.center[i])+'\n')
        file.close()

        for i in range(self.k):
            R = self.res[:,0]
            G = self.res[:,1]
            B = self.res[:,2]
            # R = R.reshape((self.img1.shape))
            # G = G.reshape((self.img1.shape))
            # B = B.reshape((self.img1.shape))
            c1 = self.center[:,0]
            c2 = self.center[:,1]
            c3 = self.center[:,2]
            # R = R.reshape((self.img1.shape))
            # G = G.reshape((self.img1.shape))
            # B = B.reshape((self.img1.shape))
            print
            R1 = np.zeros_like(R)
            R2 = np.zeros_like(G)
            R3 = np.zeros_like(B)

            for a in range(len(R)):
                if R[a] == c1[i] and G[a] == c2[i] and B[a] == c3[i]:
                    R1[a] = R[a]
                    R2[a] = G[a]
                    R3[a] = B[a]

            R1 = R1.reshape((self.img1.shape))
            R2 = R2.reshape((self.img1.shape))
            R3 = R3.reshape((self.img1.shape))


            pic = cv2.merge((R1,R2,R3))

            im = Image.fromarray(pic)
            im.save(os.path.join(self.path , 'cluster'+ str(i) + '.png'))
            # cv2.imwrite('fresher'+str(i) +'.jpeg',pic)
            self.mask(os.path.join(self.path , 'cluster'+ str(i) + '.png'),i)


    def acerage(self,arr):
        return np.count_nonzero(arr!=0)

    def segment(self):
        print(self.clusters)
        # for i in range(k):
        #     self.mask(self.clusters[i],i)

    def tiffer(self,cls):

        R = np.zeros_like(self.img1)
        G = np.zeros_like(self.img1)
        B = np.zeros_like(self.img1)

        for i in range(len(self.result)):
            for j in range(len(self.result[0])):
                    R[i][j]=self.result[i][j][0]
                    G[i][j]=self.result[i][j][1]
                    B[i][j]=self.result[i][j][2]

        something  = cv2.merge((R,G,B))

        # file = open('something.txt','w')
        # for i in range(len(something)):
        #     file.write(str(something[i])+'\n')
        # file.close()

        # print(self.result.shape,'asdsf')
        # print(self.result)
        # print('img1')
        # print(self.img1)

        for i in range(len(R)):
            for j in range(len(R[0])):
                if R[i][j] == 0 and G[i][j] == 0 and B[i][j] == 0 :
                    R[i][j] = self.img1[i][j]
                    G[i][j] = self.img2[i][j]
                    B[i][j] = self.img3[i][j]
        rgb = cv2.merge((B,G,R))
        cv2.imwrite(os.path.join(self.path ,self.name + 'rgb' + str(cls) + '.png'),rgb)
        # cv2.imwrite(os.path.join(self.path ,self.name + 'r-g-b' + str(cls) + '.tif'),rgb)
        np.putmask(R,R==0,self.bandOneMin)
        np.putmask(R,R==255,self.bandOneMax)

        np.putmask(G,G==0,self.bandTwoMin)
        np.putmask(G,G==255,self.bandTwoMax)

        np.putmask(B,B==0,self.bandThreeMin)
        np.putmask(B,B==255,self.bandThreeMax)

        rows = R.shape[1]
        cols = R.shape[0]
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(os.path.join(self.path , 'cluster'+ str(cls)) + '.tif', rows, cols, 3, gdal.GDT_UInt16)
        outdata.SetGeoTransform(self.dataset.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.dataset.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(R)
        outdata.GetRasterBand(2).WriteArray(G)
        outdata.GetRasterBand(3).WriteArray(B) ##if you want these values transparent
        # print((rgb.shape))
        # im = Image.fromarray(rgb,'I')
        # im.save(self.name + 'rgb' + str(cls) + 'pillow.png')
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def mask(self,imagename,k):
        print('masking',imagename)
        img = cv2.imread(imagename,0)
        edges = cv2.blur(img,(1,1))
        edges = cv2.Canny(edges,600,600)

        R = np.zeros_like(edges)
        G = np.zeros_like(edges)
        B = np.zeros_like(edges)


        b1 = np.zeros_like(img)
        b2 = np.zeros_like(img)
        b3 = np.zeros_like(img)

        np.putmask(b1,img!=0,255)
        np.putmask(b2,img!=0,255)

        b = cv2.merge((b1,b2,b3))

        np.putmask(R,edges!=0,255)
        np.putmask(B,edges!=0,255)
        np.putmask(G,edges!=0,255)

        edges = cv2.merge((R,G,B))
        # print('edges')
        # print(edges.shape)
        # print(edges)
        # edges = edges.reshape(self.img1.shape)

        # final = cv2.imread(imagename)
        # print('final')
        # print(final)
        #
        # np.putmask(edges,edges==0,final)

        edges = np.bitwise_or(b,edges)
        # cv2.imwrite(segment,edges)
        img = Image.fromarray(edges)
        img.save(imagename)
        self.result = edges
        self.tiffer(k)


    def kmeans(self,K):
        print('Applying Kmeans for k = ',str(K))
        self.k = K
        height , width = self.img1.shape[0] , self.img1.shape[1]
        print('height - ',height , ' width - ',width)

        X1 = self.img1.reshape((-1, 1))
        X2 = self.img2.reshape((-1, 1))
        X3 = self.img3.reshape((-1, 1))

        # X4 = np.zeros_like(X3)
        # X5 = np.zeros_like(X3)
        # value = 0
        # for i in range(len(X4)):
        #     X4[i] = value
        #     value = (value+1)%width
        # value = 0
        # for i in range(height):
        #     for j in range(width):
        #         X5[value] = i
        #         value += 1

        self.reshaped = np.hstack((X1,X2,X3))
        print('APPLYING KMEANS')
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # self.ret,self.label,self.center=cv2.kmeans(self.reshaped,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        clt = MiniBatchKMeans(K)

        self.label = clt.fit_predict(self.reshaped)
        self.res = clt.cluster_centers_.astype("uint8")[self.label]

        self.center = np.uint8(clt.cluster_centers_)

        self.center = np.uint8(self.center)
        self.res = self.center[self.label.flatten()]
        for i in range(self.k):
            for j in range(len(self.res)):
                # print(self.res[j],'compare',self.center[i])
                # print(self.res)
                if(self.center[i][0]==self.res[j][0] and self.center[i][1]==self.res[j][1] and self.center[i][2]==self.res[j][2]):
                    self.res[j][0] = self.color[i][0]
                    self.res[j][1] = self.color[i][1]
                    self.res[j][2] = self.color[i][2]

        print('Inertia - ',clt.inertia_)
        objective.append(clt.inertia_)
        print('KMEANS COMPLETED!')
        # print(self.res)
        # cv2.imwrite('resimage.jpeg',self.res)
        R = self.res[:,0]
        G = self.res[:,1]
        B = self.res[:,2]
        R = R.reshape((self.img1.shape))
        G = G.reshape((self.img1.shape))
        B = B.reshape((self.img1.shape))

        pic = cv2.merge((R,G,B))
        im = Image.fromarray(pic)
        im.save('batcluster'+str(self.k)+sys.argv[2].split('.')[0]+'.png')
        # cv2.imwrite('fresher.jpeg',pic)
        print('fresh complete')
        # print(self.ObjectiveFunction(self.k,np.uint16(self.reshaped),self.label,self.center))

        # lbl = self.label
        # self.result = lbl.reshape((self.img1.shape))
        # print(newcluster.shape)
        # print(self.res.shape)
        # orgsize = self.res.reshape((self.img1.shape))
        # print(self.img1)
        # image = self.res.reshape((self.img1.shape))
        # cv2.imwrite('resimage.tif',pic)

        # cv2.imwrite('k-output-'+str(self.k)+'.jpeg',newcluster)








if __name__ == "__main__":
    k = 2
    if len(sys.argv) >= 2:
        k = int(sys.argv[1])
    for x in range(1,k+1):
        kmeans = KMEANS()
        kmeans.loadImage(sys.argv[2])
        imagename = sys.argv[2].split('.')[0]
        kmeans.kmeans(x)
        # kmeans.seperate()
        kmeans = None

    print("Inertia - ",objective)
    plt.plot(objective)
    plt.savefig('Objective analysis.png')
    print('DONE!!!')
