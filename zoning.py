import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import matplotlib.dates as mdates
#import matplotlib.ticker as ticker
import matplotlib.cm as cm
#from numpy.random import rand
import numpy.linalg as la
from scipy.misc import imsave
import scipy.cluster.vq as vq
import scipy.stats as st

    
def cube2flat(iData):
    '''reshape 3D cube of data into 2D matrix:
    (rows - number of layers in 3D, columns - all pixels in each layer)'''
    r, h, w = iData.shape
    iData = iData.reshape(r, h*w)
    #find indeces of valid data (data are not nan on ALL records)
    notNanDataI = np.isfinite(iData.sum(axis=0))
    return iData, notNanDataI
    
    
def pca(iData, pcNumber=3, oPrefix='test', addXYGrid=False):
    '''Run principal component analysis on 3D cube of time series'''
    print 'Run PCA'
    records, height, width = iData.shape
    
    #append X,Y grids
    if addXYGrid:
        yGrid, xGrid = np.meshgrid(np.arange(0, width), np.arange(0, height))
        yGrid.shape = (1, height, width)
        xGrid.shape = (1, height, width)
        iData = np.append(iData, yGrid, 0)
        iData = np.append(iData, xGrid, 0)
        records, height, width = iData.shape

    #reshape 3D cube of data into 2D matrix and get indeces of valid pixels
    iData, notNanDataI = cube2flat(iData)
    #perform PCA on valid data
    pcaData = PCA(iData[:, notNanDataI].astype('f8').T)
    #create and fill output 2D matrix with PCA values for valid pixels
    oData = np.zeros((pcNumber, width*height), 'f4') + np.nan
    oData[:, notNanDataI] = pcaData.Y.T[0:pcNumber, :]
    #reshape 2D into 3D with individual PCAs in each layer
    oData = oData.reshape(pcNumber, height, width)
    
    # visualize PC variance
    plt.plot(pcaData.fracs, 'o-')
    plt.title('variance of PC')
    plt.savefig(oPrefix + '_pca_var.png')
    plt.close()
    
    #visualize individual PCs
    for pcn in range(0, pcNumber):
        plt.imsave('%spca%03d.png' % (oPrefix, pcn), oData[pcn, :, :])
    
    #visualize  3 PCs as RGB
    vData = np.array(oData[0:3, :, :])
    vData[np.isnan(vData)] = 0
    imsave(oPrefix + 'pca123.png', vData)

    return oData
    
    
def kmeans(iData, clustNumber, oPrefix, norm=False):
    '''Perform k-means cluster analysis and return MAP of zones'''
    print 'Run K-Means'
    
    height, width = iData.shape[1:3]
    #reshape 3D cube of data into 2D matrix and get indeces of valid pixels
    iData, notNanDataI = cube2flat(iData)
    if norm:
        #center and norm
        iDataMean = iData[:, notNanDataI].mean(axis=1)
        iDataStd  = iData[:, notNanDataI].std(axis=1)
        iData = np.subtract(iData.T, iDataMean).T
        iData = np.divide(iData.T, iDataStd).T

    #perform kmeans on valid data and return codebook
    codeBook = vq.kmeans(iData[:, notNanDataI].astype('f8').T, clustNumber)[0]
    #perform vector quantization of input data uzing the codebook
    #return vector of labels (for each valid pixel)
    labelVec = vq.vq(iData[:, notNanDataI].astype('f8').T, codeBook)[0]+1
    #create and fill MAP of zones
    zoneMap = np.zeros(width*height) + np.nan
    zoneMap[notNanDataI] = labelVec
    zoneMap = zoneMap.reshape(height, width)
    
    #visualize map of zones
    plt.imsave(oPrefix + 'zones.png', zoneMap)
    
    return zoneMap
   

def timeseries(iData, zoneMap):
    '''
    Make zone-wise averaging of input data
    input: 3D matrix(Layers x Width x Height) and map of zones (W x H)
    output: 2D matrices(L x WH) with mean and std 
    '''
    #reshape input cube into 2D matrix
    r, h, w = iData.shape
    iData, notNanDataI = cube2flat(iData)
    #get unique values of labels
    uniqZones = np.unique(zoneMap)
    # leave only not-nan
    uniqZones = uniqZones[~np.isnan(uniqZones)]
    zoneNum = np.zeros((r, uniqZones.size))
    zoneMean = np.zeros((r, uniqZones.size))
    zoneStd = np.zeros((r, uniqZones.size))
    #in each zone: get all values from input data get not nan data average
    for i in range(uniqZones.size):
        zi = uniqZones[i]
        if not np.isnan(zi):
            zoneData = iData[:, zoneMap.flat == zi]
            zoneNum[:, i] = zi
            zoneMean[:, i] = st.nanmean(zoneData, axis=1)
            zoneStd[:, i] = st.nanstd(zoneData, axis=1)
        
    return zoneMean, zoneStd, zoneNum
    
    
def hotelling(data1, data2):
    '''
    Estimate Multivariate Student T2 test (Hotelling test)
    input: dataset1, dataset2
    output: T = (m1 - m2)' x COV x (m1 - m2)
    http://en.wikipedia.org/wiki/Student%27s_t-test#Multivariate_testing
    '''
    #calculate mean, covariance and number of measurements in datsets
    mean1 = data1.mean(axis=1)
    cov1 = np.cov(data1)
    n1 = data1.shape[1]
    mean2 = data2.mean(axis=1)
    cov2 = np.cov(data2)
    n2 = data2.shape[1]
    
    #calculate total covariance (COV)
    cov12 = (n1*cov1 + n2*cov2)
    #perfrom test (m1 - m2)' x COV x (m1 - m2)
    d2 = np.dot(np.dot(mean1-mean2, la.inv(cov12)), mean1-mean2)
    #normalize by number of records in each dataset
    t2 = n1*n2*d2/(n1+n2)
    #return test value
    #for n1+n1 > 100, t2>3 gives significat difference with p=0.01
    return t2

    
def t2_test(iData, zoneMap):
    '''Build matrix of pairwise T2-test of zones'''
    #reshape input cube into 2D matrix
    r, h, w = iData.shape
    iDataF, notNanDataI = cube2flat(iData)
    nanDataI = np.isnan(iDataF.sum(axis=0))
    #get unique values of labels
    uniqZones = np.unique(zoneMap)
    #get indeces of non-nan zones
    zoneMapFlat = zoneMap.flatten()
    zoneMapFlat[nanDataI] = 0
    #create empty matrix for test values
    tMatrix = np.zeros((len(uniqZones),len(uniqZones)))
    #for all zones, perfrom the test pairwise
    for zi1 in uniqZones[1:]:
        #get non-nan data from all layers from the zone 1
        zoneData1 = iDataF[:, zoneMapFlat == zi1]
        for zi2 in uniqZones[zi1+1:]:
            print 'zi1, zi2', zi1, zi2
            #get non-nan data from all layers from the zone 2
            zoneData2 = iDataF[:, zoneMapFlat == zi2]
            try:
                #compare zones and fill test-matrix
                hotel = hotelling(zoneData1, zoneData2)
            except:
                hotel = np.nan

            tMatrix[zi1, zi2] = hotel
    
    return tMatrix
    
def plot_timeseries(iData, iDate, iDataStd=None, vData=None, figFileName=None, monthInt=1, figSize=(6,6), dpi=150, style='o-'):
    '''Make nice plots of timeseries with labels
    input:
    iData - 2D matrix (WIDTH-number of zones, HEIGHT-number
    of time steps)
    iDate - vector of dates (list of datetime objects)
    iDataStd - matrix of standard deviations (optional)
    '''
    
    #set locations and format of X-laxis tics
    months = mdates.MonthLocator(interval=monthInt)
    monthsFmt = mdates.DateFormatter('%m.%y')
    #plot all values
    fig = plt.figure(figsize=figSize, dpi=dpi)
    ax = fig.add_subplot(111)
    print iData.shape
    
    #get the same colors as in the zone map
    cmap = cm.ScalarMappable(cmap='jet')
    colors = cmap.to_rgba(np.linspace(0, 1, iData.shape[1]+1))
    
    for zn in range(0, iData.shape[1]):
        if vData is None:
            if iDataStd is None:
                ax.plot(iDate, iData[:, zn], style, color=(colors[zn+1, :3]))
            else:
                ax.errorbar(iDate, iData[:, zn], iDataStd[:, zn], fmt='o-', color=(colors[zn+1, :3]))
        else:
            X = np.arange(0, iData.shape[0])
            Y = zn
            U = iData[:, zn]
            V = vData[:, zn]
            print X, Y, U, V
            ax.quiver(X, Y, U, V, color=(colors[zn+1, :3]))

    # locate and format the tics
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    fig.autofmt_xdate()
    
    if vData is not None:
        ax.set_xlim(-1, iData.shape[0]+1)
        ax.set_ylim(-1, iData.shape[1]+1)
    
    if figFileName is None:
        plt.show()
    else:
        plt.savefig(figFileName)
        plt.close()


def average_data(iData, iDate, iYears, iMonths):
    '''Average input data over given period
    input:
    3D datacube (RxWxH, R-number of layers, W-width, H-height),
    vector of dates (list of datetime objects)
    list of Years
    list of months
    
    output:
    one 2D matrix with averaged values in each pixel
    datetime of the first data in the average
    
    usage:
    to find multi-annual monthly mean:
    averagedData = average_data(iData, iDate, [1998:2011], [7]):
    to find seasonal mean in one year
    averagedData = average_data(iData, iDate, [1998], [5:9]):
    '''
    #list of tuples (year, month) for all input dates
    yearmonth = np.array([[y.year,y.month] for y in iDate])
    
    r, h, w = iData.shape
    #create and fill temporary 3D matrix with data for averaging
    iData4aver = None
    for iy in iYears:
        for im in iMonths:
            #find appropriate layers in input data
            iDataSubset = iData[(yearmonth[:, 0] == iy) * (yearmonth[:, 1] == im), :, :]
            #append to the temporary 3D matrix
            if iData4aver is None:
                iData4aver = iDataSubset
            else:
                iData4aver = np.append(iData4aver, iDataSubset, axis=0)
            
    #average
    oDate = dt.date(iYears[0], iMonths[0], 1)
    if iMonths[0] > 10:
        oDate = dt.date(iYears[0]-1, iMonths[0], 1)
    return st.nanmean(iData4aver, axis=0).reshape(1,h,w), oDate
