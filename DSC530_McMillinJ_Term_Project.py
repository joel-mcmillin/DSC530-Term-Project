# -*- coding: utf-8 -*-
"""
Joel McMillin
DSC530 - Final Project
November 20, 2021
"""

from __future__ import print_function, division

%matplotlib inline

import numpy as np

import csv

import pandas as pd

import thinkplot
import thinkstats2


dfnyc1 = pd.read_csv(r'nyc2.csv')


''' Room Type '''

rmtp = dfnyc1
#all room types

hist = thinkstats2.Hist(rmtp.room_type, label='Room Type')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
#This plot shows that the majority of AirBnB customers selected an entire home/apt, followed by
#private room, shared room and finally hotel room. While AirBnB is popular for renting a private
#residence or space, hotels have many other ways to book, which could account for why it's such
#a low count, with such high availability. 

mean = rmtp.room_type.mean()
var = rmtp.room_type.var()
std = rmtp.room_type.std()

mean, var, std

'''
The mode of room types is Entire Home, as this is a categorical variable
While the data appears right skewed, this is due to the arbitrary order in which I
assigned the different room types to numeric data where:
    1 = entire home/apartment
    2 = private room within a residence
    3 = hotel room
    4 = shared room
'''




''' Price '''

rmtp = dfnyc1
#all room types

hist = thinkstats2.Hist(rmtp.price, label='Price')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
#Similar to the number of reviews, the pricing is also skewed toward lower 
#pricing with a higher count
hist.Largest()
'''(4000, 1)'''

mean = rmtp.price.mean()
var = rmtp.price.var()
std = rmtp.price.std()

mean, var, std
'''
(151.60202973448676, 28301.967784061133, 168.23188694198592)

Mean Price: $151.60
Variance of Price: $28,301.97
Standard Deviation of Price: $168.23
Heavily right skewed
Spread: $10 to $4,000
'''





''' Number of Reviews '''

rmtp = dfnyc1
#all room types

hist = thinkstats2.Hist(rmtp.num_revs, label='# Reviews')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
hist.Largest()
'''(1006, 1)'''

mean = rmtp.num_revs.mean()
var = rmtp.num_revs.var()
std = rmtp.num_revs.std()

mean, var, std
'''
(27.295629820051413, 2761.77577982138, 52.55260012426959)

Mean Number of Reviews: 27.3
Variance of Number of Reviews: 2762.8
Standard Deviation of Number of Reviews: 52.5
Heavily right skewed
Spread: 1 to 1006
'''





''' Last Review Date '''

dfnyc1 = pd.read_csv(r'nyc5.csv')

rmtp = dfnyc1
#all room types

hist = thinkstats2.Hist(rmtp.last_review, label='Last Review')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
#Shows 'entire home' rentals over time, from january 2016-september 2021 in NYC. 
#Count by month
''' Last Review - Count - Months Need Conversion '''
'''
Last Reviews are heavily left skewed
Spread: January 2019 to September 2021
Shows a steady rise throughout 2019 followed by a sharp drop at the beginning of 2020
and the rate of reviews doesn't increase above pre-Covid levels until early 2021, and then 
increases dramatically in summer 2021
'''
mean = rmtp.last_review.mean()
var = rmtp.last_review.var()
std = rmtp.last_review.std()

mean, var, std
'''
(43875.75661910647, 365637.8724165702, 604.6799752071919)
Mean for month of last review: February-March of 2020
Variance for month of last review: 605 months
Standard Deviation for month of last review: Roughly 21 months
'''



''' Reviews per Month '''

rmtp = dfnyc1
#all room types

hist = thinkstats2.Hist(rmtp.reviews_per_month, label='Reviews Per Month')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
#I am not discarding outliers for this b/c even though only 3 properties of any
#type have more than 150 reviews in a month average, they're notable for the fact 
#that they are from August and September 2021

mean = rmtp.reviews_per_month.mean()
var = rmtp.reviews_per_month.var()
std = rmtp.reviews_per_month.std()

mean, var, std
'''
(1.5001063550912195, 16.881598992508266, 4.108722306570288)

Mean Reviews per Month: 1.5
Variance of Reviews per Month: 16.88
Standard Deviation of Reviews per Month: 4.1
Heavily right skewed
Spread: 0.02 to 163.19
'''




''' Using a PMF to compare 2 scenarios for one variable '''

rmtp = dfnyc1

pmf = thinkstats2.Pmf(rmtp.price, label = 'Price')

thinkplot.Pmf(pmf)
thinkplot.Config(xlabel='Price', ylabel='PMF')




rmtp2 = dfnyc1[dfnyc1.room_type == 1]
#room type is entire home 
 
pmf2 = thinkstats2.Pmf(rmtp2.price, label='Price')

thinkplot.Pmf(pmf2)
thinkplot.Config(xlabel='NYC - Entire Home', ylabel='PMF')



rmtp3 = dfnyc1[dfnyc1.room_type == 2]
#room type is entire home 
 
pmf3 = thinkstats2.Pmf(rmtp3.price, label='Price')

thinkplot.Pmf(pmf3)
thinkplot.Config(xlabel='NYC - Private Room', ylabel='PMF')




thinkplot.PrePlot(2, cols = 2)
thinkplot.Hist(pmf, align = 'right')
thinkplot.Hist(pmf2, align = 'left')
thinkplot.Config(xlabel = 'Price', 
                 ylabel = 'Probability',
                 axis = [0, 350, 0, .06])


thinkplot.PrePlot(2, cols = 2)
thinkplot.Hist(pmf, align = 'right')
thinkplot.Hist(pmf2, align = 'left')
thinkplot.Config(xlabel = 'Price', 
                 ylabel = 'Probability',
                 axis = [0, 275, 0, .06])





''' CDF for a Variable '''

cdf = thinkstats2.Cdf(rmtp.num_revs, label = 'Number of Reviews')
thinkplot.Cdf(cdf)
thinkplot.Config(xlabel = 'Number of Reviews', 
                 ylabel = 'CDF', loc = 'lower right')

cdf.Prob(200)
''' 98% probability of having 200 or fewer reviews'''

cdf.Prob(150)
''' 95% probability of having 150 or fewer reviews'''

cdf.Prob(100)
''' 91% probability of having 100 or fewer reviews'''

cdf.Prob(75)
''' 87% probability of having 75 or fewer reviews'''

cdf.Prob(50)
''' 81% probability of having 50 or fewer reviews'''

cdf.Prob(30)
''' 73% probability of having 30 or fewer reviews'''
#The above tells me the probability that one will have 
#x- number of reviews or fewer. 


cdf2 = thinkstats2.Cdf(rmtp.price, label = 'Price')
thinkplot.Cdf(cdf2)
thinkplot.Config(xlabel = 'Price', 
                 ylabel = 'CDF', loc = 'lower right')

cdf2.Prob(200)
''' 82% probability of price below $200'''

cdf2.Prob(150)
''' 68% probability of price below $150'''

cdf2.Prob(100)
''' 47% probability of price below $100'''

cdf2.Prob(250)
''' 88% probability of price below $250'''

cdf2.Prob(300)
''' 92% probability of price below $300'''

cdf2.Prob(400)
''' 95% probability of price below $400'''
#The above shows information on price ranges in NYC Airbnbs. There is a 47% probability that
#the price will be below $100/night, while at $400/night there's a 95% probability of finding
#an Airbnb property (of any type).



entire_cdf = thinkstats2.Cdf(rmtp2.price, label = 'Entire Home')
prv_rm_cdf = thinkstats2.Cdf(rmtp3.price, label = 'Private Room')

thinkplot.PrePlot(2)
thinkplot.Cdfs([entire_cdf, prv_rm_cdf])
thinkplot.Show(xlabel = 'price', ylabel = 'CDF')
#This comparison shows that private room pricing is usually lower than pricing for entire residences
''' Price (Entire Room v. Private Room) '''




''' Plotting 1 Analytical Distribution with Analysis of Application to My Dataset '''
# Normal ; Normal Prob ; Log ; Pareto ; 

import scipy.stats

dfnyc1 = pd.read_csv(r'nyc2.csv')

prices = dfnyc1.price.dropna()


#Normal Distribution: 
    
mu, var = thinkstats2.TrimmedMeanVar(prices, p=0.01)
print('Mean, Var', mu, var)
'''
Mean = 141.08
Variance = 11,816.85
'''

sigma = np.sqrt(var)
print('Sigma', sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)
'''
Sigma = 108.71
'''

thinkplot.Plot(xs, ps, label='model', color='0.6')

cdf = thinkstats2.Cdf(prices, label='data')

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf) 
thinkplot.Config(title='Prices',
                 xlabel='Prices',
                 ylabel='CDF',
                 loc = 'lower right')
''' Prices '''


#Normal Probability Plot below:

mean, var = thinkstats2.TrimmedMeanVar(prices, p = 0.01)
std = np.sqrt(var)

xs = [-5, 5]
fxs, fys = thinkstats2.FitLine(xs, mean, std)
thinkplot.Plot(fxs, fys, linewidth = 4, color = '0.8')

xs, ys = thinkstats2.NormalProbability(prices)
thinkplot.Plot(xs, ys, label = 'All Property Types')

thinkplot.Config(title = 'Normal Probability Plot', 
                 xlabel = 'Standard Deviation from Mean',
                 ylabel = 'Prices')
''' Normal Probability Plot '''


#Lognormal Model

dfnyc1 = pd.read_csv(r'nyc2.csv')

price = dfnyc1.price.dropna()

def MakeNormalModel(price):
    ''' Plots a CDF with a Normal model '''
    
    cdf = thinkstats2.Cdf(price, label='Price')

    mean, var = thinkstats2.TrimmedMeanVar(price)
    std = np.sqrt(var)
    print('n, mean, std', len(price), mean, std)

    xmin = mean - 4 * std
    xmax = mean + 4 * std

    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)
    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')
    thinkplot.Cdf(cdf)
''' Price '''


MakeNormalModel(price)
thinkplot.Config(title='Price', xlabel='Prices',
                 ylabel='CDF', loc='lower right')
''' Log Scale Prices '''
#The Lognormal plot is a much better fit for showing pricing


log_price = np.log10(price)
MakeNormalModel(log_price)
thinkplot.Config(title='Log Scale Prices', xlabel='Prices',
                 ylabel='CDF', loc='upper right')
''' Prices - Log Scale '''


#Normal Probability Plot 

def MakeNormalPlot(price):
    ''' Generates a normal probability plot of birth weights '''
    
    mean, var = thinkstats2.TrimmedMeanVar(price, p=0.01)
    std = np.sqrt(var)

    xs = [-5, 5]
    xs, ys = thinkstats2.FitLine(xs, mean, std)
    thinkplot.Plot(xs, ys, color='0.8', label='model')

    xs, ys = thinkstats2.NormalProbability(price)
    thinkplot.Plot(xs, ys, label='prices')

MakeNormalPlot(price)
thinkplot.Config(title='Normal Plot Prices', xlabel='Prices',
                 ylabel='CDF', loc='lower right')
''' Normal Plot Prices '''


MakeNormalPlot(log_price)
thinkplot.Config(title = 'Lognormal Prices',
                 xlabel = 'Prices',
                 ylabel = 'CDF', 
                 loc = 'lower right')
''' Prices - Lognormal Plot '''





''' Two Scatter Plots Comparing Variables with Analysis '''

# two plots, two variables
# analysis on correlation/causation
#  - consider: covariance // pearson's correlation // non linear relationships


''' Price - Last Review '''

df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)
df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
last_review, price = sample.last_review, sample.price

thinkplot.Scatter(last_review, price, alpha=1)
thinkplot.Config(xlabel='Last Review',
                 ylabel='Price',
                 legend=False)
''' Plot: Price vs Time (last_review) '''



''' Revs per Month - Last Review '''

df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)
df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
last_review, reviews_per_month = sample.last_review, sample.reviews_per_month

thinkplot.Scatter(last_review, reviews_per_month, alpha=1)
thinkplot.Config(xlabel='Last Review',
                 ylabel='Reviews per Month',
                 legend=False)
''' Plot: Reviews per Month vs Last Review '''



''' Reviews per Month - Price '''

df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)
df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
price, reviews_per_month = sample.price, sample.reviews_per_month

thinkplot.Scatter(price, reviews_per_month, alpha=1)
thinkplot.Config(xlabel='Price',
                 ylabel='Reviews per Month',
                 legend=False)
''' Plot: Reviews per Month vs Price '''



''' Number of Reviews - Price '''

df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)
df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
price, num_revs = sample.price, sample.num_revs

thinkplot.Scatter(price, num_revs, alpha=1)
thinkplot.Config(xlabel='Price',
                 ylabel='Number of Reviews',
                 legend=False)
''' Plot: Number of Reviews - Price '''



''' Number of Reviews - Price '''

df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)
df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
num_revs, price = sample.num_revs, sample.price

thinkplot.Scatter(num_revs, price, alpha=1)
thinkplot.Config(xlabel='Number of Reviews',
                 ylabel='Price',
                 legend=False)
''' Plot: Price - Number of Reviews '''



''' Price - Reviews per Month '''

#df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)

df = pd.read_csv(r'nyc5.csv', nrows = None) #used to change dates to a numeric range
# that numpy can work with

df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
reviews_per_month, price = sample.reviews_per_month, sample.price

thinkplot.Scatter(reviews_per_month, price, alpha=1)
thinkplot.Config(xlabel='Reviews per Month',
                 ylabel='Price',
                 legend=False)
''' Plot: Price vs Reviews per Month '''



### COVARIANCE ###

cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])

def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov

''' Price - Last Review '''
price, last_review = cleaned.price, cleaned.last_review
Cov(price, last_review)
''' 8034.23 '''


''' Reviews per Month - Price '''
revs_per_month, price = cleaned.reviews_per_month, cleaned.price
Cov(revs_per_month, price)
''' 3.68 '''


''' Price - Reviews per Month '''
Cov(price, revs_per_month)
''' 3.68 '''


''' Price - Number of Reviews '''
price, num_revs = cleaned.price, cleaned.num_revs
Cov(price, num_revs)
''' -360.4 '''

''' Number of Reviews - Last Review '''
num_revs, last_review = cleaned.num_revs, cleaned.last_review
Cov(num_revs, last_review)
''' 8736.78 '''



### PEARSON'S CORRELATION ###

def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

''' Price - Last Review '''
Corr(price, last_review)
''' 0.0789 '''
np.corrcoef(price, last_review)
''' 0.0789 '''


''' Review per Month - Price '''
Corr(revs_per_month, price)
''' 0.0053 '''
np.corrcoef(revs_per_month, price)
''' 0.0053 '''


''' Price - Number of Reviews '''
Corr(price, num_revs)
''' -0.038 '''
np.corrcoef(price, num_revs)
''' -0.038 '''


''' Number of Reviews - Last Review '''
Corr(num_revs, last_review)
''' 0.259 '''
np.corrcoef(num_revs, last_review)
''' 0.259 '''


### SPEARMANS CORRELATION ###

import pandas as pd

def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)


''' Price - Last Review '''
SpearmanCorr(price, last_review)
''' 0.13145 '''


''' Reviews per Month - Price '''
SpearmanCorr(revs_per_month, price)
''' 0.08073 '''


''' Price - Number of Reviews '''
SpearmanCorr(price, num_revs)
''' 0.02297 '''


''' Number of Reviews - Last Review '''
SpearmanCorr(num_revs, last_review)
''' 0.34051 '''


''' Reviews per Month - Last Review '''
SpearmanCorr(revs_per_month, last_review)
''' 0.7346 '''





''' Chapter 9 - Hypothesis Testing ''' 

class HypothesisTest(object):

    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def PValue(self, iters=1000):
        self.test_stats = [self.TestStatistic(self.RunModel()) 
                           for _ in range(iters)]

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def TestStatistic(self, data):
        raise UnimplementedMethodException()

    def MakeModel(self):
        pass

    def RunModel(self):
        raise UnimplementedMethodException()

class CoinTest(HypothesisTest):

    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def RunModel(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice('HT') for _ in range(n)]
        hist = thinkstats2.Hist(sample)
        data = hist['H'], hist['T']
        return data

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
  

### Testing Correlation ###  

class CorrelationPermute(thinkstats2.HypothesisTest):

   def TestStatistic(self, data):
       xs, ys = data
       test_stat = abs(thinkstats2.Corr(xs, ys))
       return test_stat

   def RunModel(self):
       xs, ys = self.data
       xs = np.random.permutation(xs)
       return xs, ys   
  

#df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)

df = pd.read_csv(r'nyc5.csv', nrows = None) #this line to convert Last Review
# to a range of numbers that can be used as opposed to the dates

df.head()
    
cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])


''' Price - Last Review '''    
data = cleaned.price.values, cleaned.last_review.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()
pvalue  
''' PValue - 0.0 ... unlikely by chance '''


''' Price - Reviews per Month '''
data = cleaned.price.values, cleaned.reviews_per_month.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()
pvalue  
''' PValue - 0.351 ... by chance '''



''' Price - Number of Reviews '''  
data = cleaned.price.values, cleaned.num_revs.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()
pvalue  
''' PValue - 0.0 ... unlikely by chance ''' 
 




''' Regression Analysis - One Dependent + One Explanatory Var '''


#SINGLE REGRESSION#

#df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)   

df = pd.read_csv(r'nyc5.csv', nrows = None)  
cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])    
    
price = cleaned.price    
num_revs = cleaned.num_revs    
revs_per_month = cleaned.reviews_per_month    
last_review = cleaned.last_review
    
from thinkstats2 import Mean, MeanVar, Var, Std, Cov

def LeastSquares(xs, ys):
    meanx, varx = MeanVar(xs)
    meany = Mean(ys)

    slope = Cov(xs, ys, meanx, meany) / varx
    inter = meany - slope * meanx

    return inter, slope    
    
inter, slope = LeastSquares(price, num_revs)
inter, slope    
''' (33.52001916104186, -0.012734722185256989) '''
    

inter, slope = LeastSquares(price, revs_per_month)
inter, slope        
''' (1.4804098845607434, 0.00012992220859425215) '''   
   
 
inter, slope = LeastSquares(num_revs, revs_per_month)
inter, slope     
''' (0.7801616462214668, 0.022790698587544997) '''

inter, slope = LeastSquares(num_revs, price)
inter, slope    
''' (155.27261864737198, -0.11619681973034132) '''


inter, slope = LeastSquares(price, last_review)
inter, slope     
''' (43832.71896145397, 0.2838857614761463) '''  
''' 43832.71 is late January 2020 '''


inter, slope = LeastSquares(num_revs, last_review)
inter, slope    
''' (43786.77562649456, 2.816798231384011) '''
''' 43786.8 is between November and December 2019 ''' 
    
    
inter, slope = LeastSquares(revs_per_month, last_review)
inter, slope    
''' (43819.915141017795, 37.225012679369414) '''
''' 43819.9 is between December 2019 and January 2020  '''  
    

def FitLine(xs, inter, slope):
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys

'''Price - Reviews per Month'''
fit_xs, fit_ys = FitLine(price, inter, slope)

thinkplot.Scatter(price, revs_per_month, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Price",
                 ylabel='Reviews per Month',
                 axis = [0, 1750, 0, 175],
                 legend=False)
'''Plot - Regression - Price_Revs Per Month'''



'''Price - Number of Reviews'''
fit_xs, fit_ys = FitLine(price, inter, slope)

thinkplot.Scatter(price, num_revs, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Price",
                 ylabel='Number of Reviews',
                 axis = [0, 2000, 0, 300],
                 legend=False)
'''Plot - Regression - Price_Number of Reviews'''



'''Price - Last Review'''
fit_xs, fit_ys = FitLine(price, inter, slope)

thinkplot.Scatter(price, last_review, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Price",
                 ylabel='Last Review',
                 axis = [0, 2000, 41000, 45000],
                 legend=False)
'''Plot - Regression - Price_Last Review'''



'''Last Review - Price'''
fit_xs, fit_ys = FitLine(last_review, inter, slope)

thinkplot.Scatter(last_review, price, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Last Review",
                 ylabel='Price',
                 axis = [41000, 45000, 0, 2500],
                 legend=False)
'''Plot - Regression - Last Review_Price '''



'''Last Review - Number of Review'''
fit_xs, fit_ys = FitLine(last_review, inter, slope)

thinkplot.Scatter(last_review, num_revs, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Last Review",
                 ylabel='Number of Reviews',
                 axis = [41000, 45000, 0, 500],
                 legend=False)
'''Plot - Regression - Last Review_Number of Reviews'''


'''Last Review - Reviews per Month'''
fit_xs, fit_ys = FitLine(last_review, inter, slope)

thinkplot.Scatter(last_review, revs_per_month, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Last Review",
                 ylabel='Reviews per Month',
                 axis = [41500, 45000, 0, 100],
                 legend=False)
'''Plot - Regression - Last Review_Reviews per Month'''


#MULTIPLE REGRESSION#

import statsmodels.formula.api as smf

#df = pd.read_csv(r'nyc2.csv', parse_dates = [4], nrows = None)   

df = pd.read_csv(r'nyc5.csv', nrows = None)  
cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])    

formula = 'num_revs ~ last_review'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:               num_revs   R-squared:                       0.067
Model:                            OLS   Adj. R-squared:                  0.067
Method:                 Least Squares   F-statistic:                     1927.
Date:                Sun, 14 Nov 2021   Prob (F-statistic):               0.00
Time:                        17:15:15   Log-Likelihood:            -1.4430e+05
No. Observations:               26703   AIC:                         2.886e+05
Df Residuals:                   26701   BIC:                         2.886e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept   -1016.8445     23.886    -42.570      0.000   -1063.663    -970.026
last_review     0.0239      0.001     43.897      0.000       0.023       0.025
==============================================================================
Omnibus:                    20753.897   Durbin-Watson:                   1.516
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           551707.839
Skew:                           3.566   Prob(JB):                         0.00
Kurtosis:                      24.095   Cond. No.                     3.18e+06
'''



formula = 'num_revs ~ price'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:               num_revs   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     39.57
Date:                Sun, 14 Nov 2021   Prob (F-statistic):           3.22e-10
Time:                        17:16:15   Log-Likelihood:            -1.4521e+05
No. Observations:               26703   AIC:                         2.904e+05
Df Residuals:                   26701   BIC:                         2.904e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     33.5200      0.458     73.114      0.000      32.621      34.419
price         -0.0127      0.002     -6.290      0.000      -0.017      -0.009
==============================================================================
Omnibus:                    20952.866   Durbin-Watson:                   1.707
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           529231.360
Skew:                           3.642   Prob(JB):                         0.00
Kurtosis:                      23.557   Cond. No.                         305.
'''



formula = 'price ~ last_review'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.006
Model:                            OLS   Adj. R-squared:                  0.006
Method:                 Least Squares   F-statistic:                     167.6
Date:                Sun, 14 Nov 2021   Prob (F-statistic):           3.21e-38
Time:                        17:18:29   Log-Likelihood:            -1.7467e+05
No. Observations:               26703   AIC:                         3.493e+05
Df Residuals:                   26701   BIC:                         3.494e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept    -812.5237     74.478    -10.910      0.000    -958.504    -666.544
last_review     0.0220      0.002     12.946      0.000       0.019       0.025
==============================================================================
Omnibus:                    33872.151   Durbin-Watson:                   1.859
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          8545704.503
Skew:                           6.884   Prob(JB):                         0.00
Kurtosis:                      89.551   Cond. No.                     3.18e+06
'''



formula = 'price ~ num_revs'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     39.57
Date:                Sun, 14 Nov 2021   Prob (F-statistic):           3.22e-10
Time:                        17:19:31   Log-Likelihood:            -1.7473e+05
No. Observations:               26703   AIC:                         3.495e+05
Df Residuals:                   26701   BIC:                         3.495e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    155.2726      1.183    131.283      0.000     152.954     157.591
num_revs      -0.1162      0.018     -6.290      0.000      -0.152      -0.080
==============================================================================
Omnibus:                    33715.928   Durbin-Watson:                   1.853
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          8295244.688
Skew:                           6.833   Prob(JB):                         0.00
Kurtosis:                      88.257   Cond. No.                         73.6
'''



formula = 'last_review ~ price'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()
'''
                      OLS Regression Results                            
==============================================================================
Dep. Variable:            last_review   R-squared:                       0.006
Model:                            OLS   Adj. R-squared:                  0.006
Method:                 Least Squares   F-statistic:                     167.6
Date:                Sun, 14 Nov 2021   Prob (F-statistic):           3.21e-38
Time:                        17:22:47   Log-Likelihood:            -2.0883e+05
No. Observations:               26703   AIC:                         4.177e+05
Df Residuals:                   26701   BIC:                         4.177e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   4.383e+04      4.966   8826.986      0.000    4.38e+04    4.38e+04
price          0.2839      0.022     12.946      0.000       0.241       0.327
==============================================================================
Omnibus:                     3403.975   Durbin-Watson:                   1.345
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4907.080
Skew:                          -1.049   Prob(JB):                         0.00
Kurtosis:                       3.105   Cond. No.                         305.
'''



formula = 'last_review ~ num_revs'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:            last_review   R-squared:                       0.067
Model:                            OLS   Adj. R-squared:                  0.067
Method:                 Least Squares   F-statistic:                     1927.
Date:                Sun, 14 Nov 2021   Prob (F-statistic):               0.00
Time:                        17:23:47   Log-Likelihood:            -2.0798e+05
No. Observations:               26703   AIC:                         4.160e+05
Df Residuals:                   26701   BIC:                         4.160e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   4.379e+04      4.109   1.07e+04      0.000    4.38e+04    4.38e+04
num_revs       2.8168      0.064     43.897      0.000       2.691       2.943
==============================================================================
Omnibus:                     2860.245   Durbin-Watson:                   1.148
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3900.753
Skew:                          -0.936   Prob(JB):                         0.00
Kurtosis:                       2.960   Cond. No.                         73.6
'''

''' While attempting Logistic Regression with the variables, the formula kept 
giving error messages for which I was unable to ascertain any solution. That said,
I am deferring to the ch. 10 option of One Dependent/One Explanatory Variable, 
which is shown above '''







