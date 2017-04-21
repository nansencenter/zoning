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
import scipy.ndimage as nd

from sklearn.cluster import KMeans

def cube2flat(iData):
    '''reshape 3D cube of data into 2D matrix:
    (rows - number of layers in 3D, columns - all pixels in each layer)'''
    r, h, w = iData.shape
    iData = iData.reshape(r, h*w).copy()
    #find indeces of valid data (data are not nan on ALL records)
    notNanDataI = np.isfinite(iData.sum(axis=0))
    return iData, notNanDataI


def pca(iData, pcNumber=3, oPrefix='test', addXYGrid=False, percentile=0.1):
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

    # clip input data to the given percentile
    for i in range(records):
        gpi = np.isfinite(iData[i, :, :])
        vmin = np.percentile(iData[i, :, :][gpi], percentile)
        vmax = np.percentile(iData[i, :, :][gpi], 100-percentile)
        iData[i, :, :][iData[i, :, :] < vmin] = vmin
        iData[i, :, :][iData[i, :, :] > vmax] = vmax

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
    plt.semilogy(pcaData.fracs, 'o-')
    plt.title('variance of PC')
    plt.savefig(oPrefix + '_pca_var.png')
    plt.close()

    #visualize individual PCs
    for pcn in range(0, pcNumber):
        gpi = np.isfinite(oData[pcn, :, :])
        vmin = np.percentile(oData[pcn, :, :][gpi], percentile)
        vmax = np.percentile(oData[pcn, :, :][gpi], 100-percentile)
        oData[pcn, :, :][oData[pcn, :, :] < vmin] = vmin
        oData[pcn, :, :][oData[pcn, :, :] > vmax] = vmax
        plt.imsave('%spca%03d.png' % (oPrefix, pcn), oData[pcn, :, :])

    #visualize  1,2,3 PCs as RGB
    cube2rgb(oPrefix + 'pca123.png', oData[0:3, :, :])

    #visualize 4,5,6 PCs as RGB
    if pcNumber >= 6:
        cube2rgb(oPrefix + 'pca456.png', oData[3:6, :, :])

    np.save(oPrefix + 'pca.npy', oData)

    # create correlation matrix figure
    corr_matrix(iData.reshape(records, height, width), oData, oPrefix)

    return oData

def cube2rgb(oFileName, iData, maxVal=3):
    vData = np.dstack(iData)
    vData[np.isnan(vData)] = 0
    vData[vData < -maxVal] = -maxVal
    vData[vData > maxVal] = maxVal
    imsave(oFileName, vData)


def kmeans(iData, clustNumber, oPrefix, norm=False, addXYGrid=False, xyFactor=1, iters=100):
    '''Perform k-means cluster analysis and return MAP of zones'''
    print 'Run K-Means'


    # get shape
    height, width = iData.shape[1:3]

    #append X,Y grids
    if addXYGrid:
        yGrid, xGrid = np.meshgrid(np.arange(0, width), np.arange(0, height))
        yGrid = (yGrid - yGrid.mean()) / yGrid.std()
        xGrid = (xGrid - xGrid.mean()) / xGrid.std()
        yGrid.shape = (1, height, width)
        xGrid.shape = (1, height, width)
        iData = np.append(iData, yGrid, 0)
        iData = np.append(iData, xGrid, 0)
        records, height, width = iData.shape

    # mask out-of-roi
    mask = iData[0]==0

    #center and norm
    if norm:
        iData = center_and_norm(iData)

    # change significance of X,Y fields
    if addXYGrid:
        iData[-1] *= xyFactor
        iData[-1] *= xyFactor


    #reshape 3D cube of data into 2D matrix and get indeces of valid pixels
    iData, notNanDataI = cube2flat(iData)

    #perform kmeans on valid data and return codebook
    #codeBook = vq.kmeans(iData[:, notNanDataI].astype('f8').T, clustNumber, iters)[0]
    #perform vector quantization of input data uzing the codebook
    #return vector of labels (for each valid pixel)
    #labelVec = vq.vq(iData[:, notNanDataI].astype('f8').T, codeBook)[0]+1

    labelVec = KMeans(n_clusters=clustNumber, n_jobs=-1).fit_predict(iData[:, notNanDataI].astype('f8').T)

    #create and fill MAP of zones
    zoneMap = np.zeros(width*height) + np.nan
    zoneMap[notNanDataI] = labelVec
    zoneMap = zoneMap.reshape(height, width)
    zoneMap[mask] = np.nan

    #visualize map of zones
    plt.imsave(oPrefix + 'zones.png', zoneMap)

    # save to numpy file
    np.save(oPrefix + 'zones.npy', zoneMap)

    return zoneMap

def center_and_norm(iData):
    r, h, w = iData.shape
    # reshape into 2D
    iData, notNanDataI = cube2flat(iData)
    # find mean, std
    iDataMean = iData[:, notNanDataI].mean(axis=1)
    iDataStd  = iData[:, notNanDataI].std(axis=1)
    # center and norm
    iData = np.subtract(iData.T, iDataMean).T
    iData = np.divide(iData.T, iDataStd).T
    # reshape into 3D
    iData.shape = (r, h, w)
    return iData

def timeseries(iData, zoneMap, std=None):
    '''
    Make zone-wise averaging of input data
    input: 3D matrix(Layers x Width x Height) and map of zones (W x H)
    output: 2D matrices(L x WH) with mean and std
    '''
    #reshape input cube into 2D matrix
    r, h, w = iData.shape
    iData, notNanDataI = cube2flat(iData)
    #get unique values of not-nan labels
    uniqZones = np.unique(zoneMap[np.isfinite(zoneMap)])
    zoneNum = np.zeros((r, uniqZones.size))
    zoneMean = np.zeros((r, uniqZones.size))
    zoneStd = np.zeros((r, uniqZones.size))
    #in each zone: get all values from input data get not nan data average
    for i in range(uniqZones.size):
        zi = uniqZones[i]
        if not np.isnan(zi):
            zoneData = iData[:, zoneMap.flat == zi]
            zoneNum[:, i] = zi
            zoneMean[:, i] = np.nanmean(zoneData, axis=1)
            zoneStd[:, i] = np.nanstd(zoneData, axis=1)
            if std is not None:
                # filter out of maxSTD values
                outliers = (np.abs(zoneData.T - zoneMean[:, i]) > zoneStd[:, i] * std).T
                zoneData[outliers] = np.nan
                zoneMean[:, i] = np.nanmean(zoneData, axis=1)
                zoneStd[:, i] = np.nanstd(zoneData, axis=1)

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

def plot_timeseries(iData, iDate, iDataStd=None, vData=None,
                    figFileName=None, monthInt=1, figSize=(6,6),
                    dpi=150, style='o-',
                    legend=None, title=None, dateFormat='%m.%y',
                    labels=None):
    '''Make nice plots of timeseries with legend and labels
    input:
    iData - 2D matrix (WIDTH-number of zones, HEIGHT-number
    of time steps)
    iDate - vector of dates (list of datetime objects)
    iDataStd - matrix of standard deviations (optional)
    '''

    #set locations and format of X-laxis tics
    months = mdates.MonthLocator(interval=monthInt)
    monthsFmt = mdates.DateFormatter(dateFormat)
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
                ax.errorbar(iDate, iData[:, zn], iDataStd[:, zn], fmt=style, color=(colors[zn+1, :3]))
            if labels is not None:
                #import ipdb; ipdb.set_trace()
                maxI = np.argmax(iData[:, zn])
                ax.text(iDate[maxI], iData[maxI, zn], labels[zn],
                        color=(colors[zn+1, :3]),
                        bbox=dict(facecolor='white',
                                    alpha=0.9,
                                    linewidth=0))
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

    if legend is not None and len(legend)==iData.shape[1]:
        plt.legend(legend)

    if title is not None:
        plt.title(title)

    plt.tight_layout(1.5)

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
    return np.nanmean(iData4aver, axis=0).reshape(1,h,w), oDate


def fill_gaps_nn(iData, size=2, badValue=None):
    ''' Fill gaps using nearest neigbour interpolation '''

    # convert data to 3D cube
    if len(iData.shape) == 2:
        procData = np.array([iData])
        twoD = True
    elif len(iData.shape) == 3:
        procData = np.array(iData)
        twoD = False
    else:
        raise Exception('Can handle only 2D or 3D data')

    # fill gaps in each layer of the 3D cube
    for i in range(len(procData)):
        # Fill gaps:
        # extrapolate valid values into 2 pixels border using nearest neighbour
        #     get distance and indices of nearest neighbours
        if badValue is None:
            mask = np.isnan(procData[i])
        else:
            mask = procData[i] == badValue
        dst, ind = nd.distance_transform_edt(mask,
                                             return_distances=True,
                                             return_indices=True)
        #     erase row,col indeces further than 2 pixels
        ind[0][dst > size] = 0
        ind[1][dst > size] = 0
        #    fill gaps
        procData[i] = procData[i][tuple(ind)]

    # if input data was 2D, return 2D also
    if twoD:
        procData = procData[0]

    return procData

def clean_zones(zones, minSegmentSize):
    ''' Remove small zones and replace with nearest neigbour'''
    # bad zones mask
    badMask = zones == -1

    # split multi-part zones
    zonesAll = split_multi_zones(zones)

    # find areas of all zones
    zAllIndeces = np.unique(zonesAll)
    zAllIndeces = zAllIndeces[zAllIndeces >= 0]
    zAllAreas = nd.sum(np.ones_like(zones), zonesAll, zAllIndeces)

    # set zones with small areas to -1
    for zai in zAllIndeces[zAllAreas < minSegmentSize]:
        zonesAll[zonesAll == zai] = -1

    # fill small segments with values from nearest neighbours
    invalid_cell_mask = zonesAll == -1
    indices = nd.distance_transform_edt(invalid_cell_mask, return_distances=False, return_indices=True)
    zonesClean = zones[tuple(indices)]

    # mask bad values with 0
    zonesClean[badMask] = -1

    return zonesClean

def split_multi_zones(zones):
    ''' Split zones which have mulitiple disconnected parts into new zones '''
    # indices of all non-zero zones
    zIndices = np.unique(zones)
    zIndices = zIndices[zIndices >= 0]
    structure = np.ones((3,3)) # for labeling

    # matrix for all spatially separate zones (-1 masks bad data)
    zonesAll = np.zeros_like(zones) - 1
    lblCounter = 0
    for zi in zIndices:
        # find zone
        mask = zones == zi
        # split spatially
        labels, nl = nd.label(mask, structure)
        # add unique numbers
        labels[labels > 0] += lblCounter
        # add zones to new matrix
        zonesAll += labels
        lblCounter += nl

    return zonesAll

def renumber_zones(zones, random=False):
    # change nambering from 0 to n_zones
    i = 0
    renZones = np.zeros_like(zones)
    rndIndeces = np.random.permutation(range(len(np.unique(zones))))
    for zi in np.unique(zones):
        newIndex = i
        if random:
            newIndex = rndIndeces[i]
        renZones[zones == zi] = i
        i += 1

    return renZones

def corr_matrix(iData, pcs=None, oPrefix='', sqr=False, show=False, dpi=150):
    if pcs is not None:
        iData = np.vstack([iData, pcs])

    iData, notNanDataI = cube2flat(iData)
    r = np.corrcoef(iData[:, notNanDataI])
    vmin = -1
    if sqr:
        r *= r
        vmin = 0
    plt.imshow(r, interpolation='nearest', vmin=vmin, vmax=1)
    plt.savefig(oPrefix + '_pca_cor.png',
                      bbox_inches='tight', pad_inches=0, dpi=dpi)
    if show:
        plt.show()

    plt.close()
