"""apop.py computes UK Muslim, non-Muslim and Jihadist population projections.


This code was written under:

Python     version 3.5.2
numpy      version 1.13.1
scipy      version 0.19.1
pandas     version 0.20.3
matplotlib version 2.0.2


This code requires the following data files be present in order to run:

age_sex_all_religions_england_wales_2011.csv
age_sex_muslims_england_wales_2011.csv
age_sex_not_stated_england_wales_2011.csv
TABLE A.5. Period fertility indicators.xlsx
lalevelasfrs.xls
deathsarea2011_tcm77-295437.xls

The sources for these data files are described in comments below.


To run this code type the following into your Python interpreter:

from apop import *
process()


All Rights Released.
This code is in the Public Domain.

This is free software; there is NO WARRANTY; not even for MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.

This code will not be supported.


Code originally authored by Paul Revere Jr.
Code originally hosted at https://github.com/paul-revere-jr/Projections
Code originally written in 2017.


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import leslie


# population data classes, filled by fill_pop_classes

class Tot: pass
class Non: pass
class Mus: pass

tot = Tot()
non = Non()
mus = Mus()


def fill_pop_classes():
    """Fill the global population classes with hand coded data"""

    # From en.wikipedia.org/wiki/Demography_of_the_United_Kingdom#Population
    # at the 2011 census the population of the countries of the UK were:
    
    tot.eng = 53012456
    tot.sco = 5295000
    tot.wal = 3063456
    tot.ni  = 1810863
    tot.uk  = 63181775
    
    # From https://en.wikipedia.org/wiki/Islam_in_the_United_Kingdom
    # The 2011 muslim population of the countries of the UK were:
    
    mus.eng = 2660116
    mus.sco = 76737
    mus.wal = 45950 
    mus.ni  = 3832 
    mus.uk  = 2786635
    
    # Country non figures can be obtained as tot-mus.
    
    non.eng = tot.eng - mus.eng
    non.sco = tot.sco - mus.sco
    non.wal = tot.wal - mus.wal
    non.ni  = tot.ni  - mus.ni
    non.uk  = tot.uk  - mus.uk
    

def load_paps():
    """Return DataFrames of UK mus and non-mus 2011 population age profiles"""


    # The following csv files were downloaded from
    # https://www.nomisweb.co.uk/census/2011/dc2107ew
    # They contain age profile population figures.
    
    df_all = pd.read_csv('age_sex_all_religions_england_wales_2011.csv',
                         skiprows=10,skipfooter=33,index_col=0,engine='python')

    df_mus = pd.read_csv('age_sex_muslims_england_wales_2011.csv',
                         skiprows=10,skipfooter=33,index_col=0,engine='python')
    
    df_not = pd.read_csv('age_sex_not_stated_england_wales_2011.csv',
                         skiprows=10,skipfooter=33,index_col=0,engine='python')

    # Stating religion on the 2011 census was optional, and 7.2 per
    # cent of people did not answer the question.  The age
    # distribution of these 'nots' is in df_not.  The following
    # calculations continue under the assumption that NONE of these
    # 'nots' were Muslims.  This is almost certainly incorrect, and so
    # the numbers in df_mus are almost certainly low, but I don't
    # attempt to make any correction for this because I don't want
    # anyone to be able to accuse these calculations of "inventing"
    # Muslims.  ALL 'not's are therefore assumed to be non-Muslims.

    # df_all includes both mus and not
    # df_non is defined here as all - mus; it therefore includes all 'nots'

    df_non = df_all - df_mus
    
    # df_mus and df_non are now the muslim and non-muslim data for
    # England and Wales for 2011

    # mus class population numbers can be used to scale up the England
    # and Wales age data above to the whole country.  The assumption
    # being made is that the Muslims present in Scotland and Northern
    # Ireland have the same age profile as those in England and Wales:

    df_mus = df_mus * mus.uk / (mus.eng + mus.wal)

    # non class population figures can be used to scale up the
    # non-Muslim age data for England and Wales to the whole country.
    # The assumption being made is that the age profile of non-Muslims
    # in Scotland and Northern Ireland is the same as the age profile
    # of non-Muslims in England and Wales.

    df_non = df_non * non.uk / (non.eng + non.wal)

    # df_mus and df_non now represent the age profiled populations of
    # Muslims and non-Muslims in the whole UK at the 2011 census.
    # However, they currently contain extra detail in the 5-19 year
    # age groups that is not required for this analysis.  This extra
    # detail is combined and removed here.

    i = pd.Series(df_non.index)
    i[1] = 'Age 5 to 9'
    i[4] = 'Age 15 to 19'
    df_non.index = i
    df_mus.index = i
    
    df_mus.loc['Age 5 to 9'] += df_mus.loc['Age 8 to 9']
    df_non.loc['Age 5 to 9'] += df_non.loc['Age 8 to 9']
    
    df_mus.loc['Age 15 to 19'] += df_mus.loc['Age 16 to 17']
    df_mus.loc['Age 15 to 19'] += df_mus.loc['Age 18 to 19']
    df_non.loc['Age 15 to 19'] += df_non.loc['Age 16 to 17']
    df_non.loc['Age 15 to 19'] += df_non.loc['Age 18 to 19']
    
    df_mus.drop(['Age 8 to 9','Age 16 to 17','Age 18 to 19'],inplace=True)
    df_non.drop(['Age 8 to 9','Age 16 to 17','Age 18 to 19'],inplace=True)
    
    # sanity check
    
    assert df_mus.sum().sum() + df_non.sum().sum() == tot.uk

    # df_mus and df_non now contain the age profiled populations of
    # Muslims and non-Muslims in the whole of the UK at the 2011
    # census in 5 year bands.

    return df_mus, df_non


def load_asfrs():    
    """Return Series of ASFRs for mus & non-mus UK (2011) and Gulf (2009)"""

    # ASFR = Age Specific Fertility Rate

    # Download Table A.5 from http://www.un.org/en/development/desa/population/publications/dataset/fertility/wfr2012/MainFrame.html as "TABLE A.5. Period fertility indicators.xlsx"

    df_un = pd.read_excel('TABLE A.5. Period fertility indicators.xlsx',
                          skiprows=[0,1],header=[1],index_col=0)

    df_sau = df_un.loc['Saudi Arabia']\
             [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]
    df_bah = df_un.loc['Bahrain']\
             [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]
    df_uae = df_un.loc['United Arab Emirates']\
             [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]
    df_oma = df_un.loc['Oman']\
             [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]
    df_kuw = df_un.loc['Kuwait']\
             [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]
    df_qat = df_un.loc['Qatar']\
             [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]

    sr_gul = df_sau.iloc[2] + df_bah.iloc[2] + df_uae.iloc[2]\
             + df_oma.iloc[2] + df_kuw.iloc[2] + df_qat.iloc[2]
    sr_gul = sr_gul / 6
    sr_gul = sr_gul.drop(['Year'])

    # sr_gul is now a Pandas Series containing the mean age specific birth
    # rate averaged over the 6 richest Gulf states in approx 2009.  These
    # states all had a higher GDP (PPP) per capita than the UK in 2016
    # according to both the IMF and the CIA, see
    # https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(PPP)_per_capita


    # Download lalevelasfrs.xls from https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/conceptionandfertilityrates/adhocs/0054942010to2013localauthorityagespecificfertilityratesasfrsenglandandwales
    # This gives age specific fertility rates for England and Wales in 2011.

    df_ew = pd.read_excel('lalevelasfrs.xls',
                          sheetname='2011R(Quin)', 
                          header=[0],
                          skiprows=[0,1,3,4],
                          skip_footer=14, 
                          index_col=2)
    
    sr_ew = df_ew.loc['England and Wales']
    tfr_ew = sr_ew.loc['TFR']

    # The UN based figures from the chart above give a figure of 1 for the 
    # '45-49' range in 2009:
    df_uk = df_un.loc['United Kingdom']\
            [['Year','15-19','20-24','25-29','30-34','35-39','40-44','45-49']]
    # However, the sr_ew figures use a 40+ final band.
    # Since I want a 45-49 band I'll use the '1' from the UN figures from 2009
    # and subtract that '1' from the 40+ value given in sr_ew:
    val = sr_ew[8]
    sr_ew = sr_ew.drop(['Unnamed: 0','Unnamed: 1','<15','Unnamed: 11',
                        'Unnamed: 12','TFR','40+'])
    sr_ew = sr_ew.append(pd.Series([val-1.0,1.0],index=['40-44','45-49']))

    # Assume that the age specific fertility rates for the UK as a whole are
    # the same as those for England and Wales.
    
    sr_uk = sr_ew

    # sr_uk contains the age specific fertility rates for the whole UK in 2011


    # Age specific fertility rates for Muslims in the UK are hard to
    # come by.  The only source I've found are the graphs in
    # sylvie-dubuc-presentation.pdf from
    # http://www.restore.ac.uk/UPTAP/wordpress/wp-content/uploads/2009/04/sylvie-dubuc-presentation.pdf
    # which has figures for people of Bangladeshi and Pakistani
    # ethnicity living in the UK.  These can be read manually from the
    # graphs:

    index  = ['15-19','20-24','25-29','30-34','35-39','40-44','45-49']
    sr_pak = pd.Series([20.0,145.0,185.0,138.0,75.0,18.0,5.0],index=index)
    # sr_pak.sum()*5/1000 yields a TFR of 2.93 for "1998-2006".

    sr_ban = pd.Series([25.0,195.0,185.0,122.0,75.0,27.0,7.0],index=index)
    # sr_ban.sum()*5/1000 yields a TFR of 3.18 for "1998-2006"

    # Government census data only seems to give TFRs (not ASFRs) for people
    # actually born abroad.  In http://www.ons.gov.uk/ons/dcp171766_350433.pdf
    # for example Table 3 gives the 2011 TFRs for Pakistan and Bangladesh-born
    # women in the UK as 3.82 and 3.25 respectively.  These are higher than
    # the figures from sylvie-dubuc-presentation.pdf because the latter include
    # births to second generation ethnic Pakistanis and Bangladeshis.  Since 
    # the latter is what we're interested in those are the figures I'll use.

    # What figures to use for the Muslim ASFRs?

    # https://en.wikipedia.org/wiki/Demography_of_the_United_Kingdom#Ethnicity
    # gives the 2011 Census data for UK residents of Pakistani and Bangladeshi
    # ethnicity as

    pak = 1173892
    ban =  451529

    # These people (with a few apostate exceptions) represent 1.6m of
    # the UK's 2.78m Muslims.  From Table 3 in dcp171766_350433.pdf
    # above it can be deduced that the remainder of the UK's Muslims
    # are mostly Indian, Nigerian, or Somalian.  Their new-immigrant
    # TFRs are given as 2.35, 3.32, and 4.19 respectively, though of
    # course Indians are probably Hindu or Sikh, and Nigerians may
    # well be Christian.  Given that, these numbers are similar enough
    # to those for Pakistan and Bangladesh in the same Table that in
    # the absense of better data it seems reasonable to assume that
    # the Pakistani and Bangladeshi ASFRs above can be used for all
    # Muslims in the UK.

    # I weight the Paskistani and Bangladeshi ASFRs appropriately:

    sr_mus = sr_pak * (pak/(pak + ban)) + sr_ban * (ban/(pak + ban))

    # sr_mus contains the age specific fertility rates for UK Muslims in 2011

    # sr_mus.sum()*5/1000 yields a Muslim TFR for the UK of 2.999.
    # I note with some satisfaction that this is practically identical to the
    # UK Muslim 2005-2010 TFR of 3.0 quoted by the Pew Research Center here:
    # http://www.pewforum.org/2011/01/27/future-of-the-global-muslim-population-regional-europe/

    # Now to compute sr_non.
    # Rearranging the relation:
    # sr_uk * tot.uk = sr_non * non.uk + sr_mus * mus.uk

    sr_non = (sr_uk * tot.uk - sr_mus * mus.uk) / non.uk

    # sr_non contains age specific fertility rates for UK non-Muslims in 2011

    # sr_non.sum()*5/1000 yields a non-Muslim TFR for the UK of 1.859.
    # This is higher than the UK non-Muslim 2005-2010 TFR of 1.8
    # quoted by the Pew Research Center on the same web page as above
    # (though the difference may be smaller than it first appears
    # since Pew only quotes 2 significant figures).

    return sr_mus, sr_non, sr_gul
    

def load_deaths():
    """Return DataFrame of age profiled UK deaths in 2011"""
    
    # Download deathsarea2011_tcm77-295437.xls from:
    # https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/deathsregisteredbyareaofusualresidenceenglandandwales

    df_ew = pd.read_excel('deathsarea2011_tcm77-295437.xls',
                          sheetname='Table 2', 
                          header=[0],
                          skiprows=[0,1,2,3,4,5],
                          skip_footer=0, 
                          index_col=0)
    
    sr_ew = df_ew.loc['ENGLAND AND WALES1']
    sr_m_ew = sr_ew.iloc[2:24:2]
    sr_f_ew = sr_ew.iloc[3:25:2]
    df_ew = pd.DataFrame({'Males':sr_m_ew.values,'Females':sr_f_ew.values},
                         index=sr_m_ew.index)
    i = pd.Series(df_ew.index)
    i[1] = '0-4'
    df_ew.index = i
    df_ew.loc['0-4'] += df_ew.loc['Under 1']
    df_ew.drop(['Under 1'],inplace=True)
    
    # df_ew now contains deaths in England and Wales in 2011 for Males
    # and Females.  The banding is a little odd being 0-4, then every
    # 10 years, and needs to be adjusted for this analysis.  The
    # adjustment will split every ten year band in half to make two 5
    # year bands.

    df = pd.DataFrame(df_ew.iloc[0]).T
    f = 0
    t = 4
    for i in range(1,df_ew.shape[0]-1):
        d = pd.DataFrame(df_ew.iloc[i])/2
        for j in range(2):
            f += 5
            t += 5
            d.columns = [str(f) + '-' + str(t)]
            df = df.append(d.T)
    df = df.append(pd.DataFrame(df_ew.iloc[df_ew.shape[0]-1]).T)
    df_ew = df

    # df_ew now contains deaths in England and Wales in 2011 for Males
    # and Females in 5 year bands.  Assume that Scotland and Northern
    # Ireland experience the same death rates as England and Wales to
    # compute df_uk:

    df_uk = df_ew * tot.uk/(tot.eng + tot.wal)

    # df_uk contains deaths in the UK in 2011 for Males and Females in
    # 5 year bands.

    return df_uk


def compute_death_rate(df_uk_deaths,df_mus,df_non):
    """Returns DataFrame of deaths per 1000 people in each band in 2011"""

    df_uk = df_non + df_mus
    df_uk.index = df_uk_deaths.index
    df_uk_dr = 1000 * df_uk_deaths / df_uk
    
    # df_uk_dr contains the number of deaths per 1000 people in each 5
    # year age band for each sex in 2011

    # Death rate can be confusing to interpret.

    # For example : (df_uk_dr['Males'] + df_uk_dr['Females'])*5/2
    # in the '0-4' row yields the UK's infant mortality rate 
    # (deaths per 1000 live births for under 5s, 4.2 in 2015 (says Wikipedia)
    # because each child spends 5 years in the band (hence *5) and
    # Males and Females are averaged to provide a figure for a single person
    # (hence the /2).  This expression yields 5.24554 as the 2011 figure.

    # A similar calculation holds for every other age band.
    # Note that the 5-year death rate in the '85 and over' age band is
    # 734/1000, which means 266 people are still alive after 5 years.
    # Unfortunately these 266 people are not captured by the Leslie
    # matrix calculations, which assume there is no survival for the
    # members of the last cell, and so these survivors must be added 
    # in manually.

    return df_uk_dr


def get_les_mats(sr_asfr,df_dr):
    """Returns M,F Leslie matrices constructed from the given parameters"""
    
    # Note that the survival array is one shorter than the fecundity array.
    # Both are fractions that apply to each 5 year cell as a unit.

    # For Males
    fec = np.zeros(18)
    sur = 1 - df_dr['Males'].values[:-1]*5/1000
    mlm = leslie(fec,sur)

    # For Females
    fec = np.double(sr_asfr.values) * 5 / 1000
    fec = np.hstack((np.zeros(3),fec,np.zeros(8)))
    sur = 1 - df_dr['Females'].values[:-1]*5/1000
    flm = leslie(fec,sur)

    return mlm, flm


def runsim(df_im,df_in,sr_im,sr_tm,sr_in,sr_tn,ytt,df_dr,pmc,eyr):
    """Returns df_m and df_n simulation results for the given parameters.

    df_im = DataFrame of initial Muslim population
    df_in = DataFrame of initial non-Muslim population
    sr_im = Series defining initial Muslim age specific fertility rate
    sr_tm = Series defining target  Muslim age specific fertility rate
    sr_in = Series defining initial non-Muslim age specific fertility rate
    sr_tn = Series defining target  non-Muslim age specific fertility rate
    ytt   = Years To Target fertility rate
    df_dr = DataFrame of death rates for all population groups
    pmc   = proportion male children for all population groups
    eyr   = End Year of the simulation


    """

    # All simulations begin in 2011 because almost all of the data
    # used was collected by the 2011 Census.  The 5 year cell sizes of
    # the age structured data dictates that the simulation will
    # advance 5 years at a time and therefore the number of simulation
    # periods is:

    simp = (eyr - 2011) // 5

    # initialise the return DataFrames

    df_m = pd.DataFrame(df_im.sum(),columns=[2011]).T
    df_n = pd.DataFrame(df_in.sum(),columns=[2011]).T

    # initialise the population arrays
    
    a_mm = df_im['Males'].values
    a_mf = df_im['Females'].values
    a_nm = df_in['Males'].values
    a_nf = df_in['Females'].values

    # compute the Male and Female survival rates for the last age cell

    df  = 1 - df_dr['Males']*5/1000
    msr = df.iloc[-1]
    df  = 1 - df_dr['Females']*5/1000
    fsr = df.iloc[-1]

    for p in range(simp):
        
        # Both the Muslim and non-Muslim ASFR Series are assumed to
        # linearly interpolate from their initial values to their
        # target values over ytt years.

        if p*5 < ytt:
            sr_m = ((ytt-(p*5))/ytt)*sr_im + (p*5/ytt)*sr_tm
        else:
            sr_m = sr_tm

        if p*5 < ytt:
            sr_n = ((ytt-(p*5))/ytt)*sr_in + (p*5/ytt)*sr_tn
        else:
            sr_n = sr_tn
            
        # get the Leslie matrices for these ASFRs

        ma_mm, ma_mf = get_les_mats(sr_m,df_dr)
        ma_nm, ma_nf = get_les_mats(sr_n,df_dr)

        # compute the numbers of final cell survivors

        mms = msr * a_mm[-1]
        mfs = fsr * a_mf[-1]
        nms = msr * a_nm[-1]
        nfs = fsr * a_nf[-1]

        # advance one time period

        a_mm = ma_mm.dot(a_mm)
        a_mf = ma_mf.dot(a_mf)
        a_nm = ma_nm.dot(a_nm)
        a_nf = ma_nf.dot(a_nf)
        
        # All the children "born" in these calculations are reported
        # in the female array's [0] position.  This is of course
        # wrong, since about half of those children will be boys.
        # They are reassigned here.

        a_mm[0] = pmc * a_mf[0]
        a_nm[0] = pmc * a_nf[0]

        a_mf[0] = (1 - pmc) * a_mf[0]
        a_nf[0] = (1 - pmc) * a_nf[0]

        # As described in compute_death_rate() some final cell members
        # survive but are lost in the Leslie matrix calculations.  This
        # happens because the last cell is "Age 85 or older".  These 
        # survivors are reinstated here.

        a_mm[-1] += mms
        a_mf[-1] += mfs
        a_nm[-1] += nms
        a_nf[-1] += nfs

        # Append the new totals to the return DataFrames

        i    = [2011 + (p+1)*5]
        df   = pd.DataFrame({'Males':a_mm.sum(),'Females':a_mf.sum()},index=i)
        df_m = df_m.append(df)
        df   = pd.DataFrame({'Males':a_nm.sum(),'Females':a_nf.sum()},index=i)
        df_n = df_n.append(df)

    return df_m, df_n



def process():
    """Top level function to perform population simulations"""

    fill_pop_classes()
    df_mus_2011, df_non_2011 = load_paps()
    sr_mus_asfr_2011, sr_non_asfr_2011, sr_gul_asfr_2009 = load_asfrs()
    df_uk_deaths_2011 = load_deaths()

    # Compute the death rate from 2011 data.
    # I will assume below that this remains constant over time.

    df_uk_dr = compute_death_rate(df_uk_deaths_2011,df_mus_2011,df_non_2011)
    
    # Compute the proportion of male children for all population groups.

    df  = df_mus_2011 + df_non_2011
    pmc = df['Males'][0] / df.iloc[0].sum()    

    # Initialise the output graph.

    fig = plt.figure(figsize=(8.27,9.3))
    ax1 = plt.axes([0.1,0.3,0.8,0.6])    

    # All simulations assume no immigration and no emmigration.  This
    # is because so many possibilities for future immigration policy
    # exist it seems pointless to choose any one or even few to model.
    # It should be obvious that if the percentage of Muslims in any
    # immigrant cohort exceeds the percentage of Muslims currently in
    # the UK, then the Islamification of the UK will be accelerated.
    # If it is significantly lower then the Islamification of the UK
    # will be retarded; the higher fecundity of new Muslim immigrants
    # also needs to be taken into account.


    # Simulation A : Convergence to UK rates.

    # This simulation assumes that the UK Muslim ASFRs will fall
    # linearly over a period of 100 years to match exactly the UK
    # non-Muslim ASFRs.  A similar 100 year convergence is used by the
    # Pew Research Center in its simulations (see p185 in [1]).  I
    # assume that the non-Muslim ASFRs remain unchanged over the
    # entire simulation period.  Other predictions for long term UK
    # TFRs exist, but I think such predictions are highly suspect and
    # I therefore prefer to stick to what we know : the current rates.
    # [1] www.pewforum.org/files/2015/03/PF_15.04.02_ProjectionsFullReport.pdf
    # [1] goo.gl/6zFh1P (shortened URL)

    df_mus, df_non = runsim(df_mus_2011,df_non_2011,sr_mus_asfr_2011,sr_non_asfr_2011,sr_non_asfr_2011,sr_non_asfr_2011,100,df_uk_dr,pmc,2500)

    sr = 100*df_mus.sum(axis=1)/(df_mus.sum(axis=1)+df_non.sum(axis=1))
    ax1.plot(sr.index,sr.values,label='Projection A')
    

    # Simulation B : Polygamy

    # This simulation assumes that UK Muslim Total Fertility Rates
    # will fall to the UK rates plus a small increment due to the
    # Muslim practice of polygamy, with ages at childbirth following
    # the distribution seen in the six Gulf states, over a period of
    # 100 years.

    # Polygamy increases birth rates by at least 3 different mechanisms:

    # 1. Polygamy increases marriage rates among women, [2].  In
    # monogamous societies there are always some men who are either
    # gay, asexual, in prison, too poor, too busy or who for some
    # other reason don't get married, and there are therefore an equal
    # number of women who can't get married either.  In a polygamous
    # society these women can marry a man who is already married.

    # 2. Polygamy creates a 'shortage' of women, and that shortage
    # tends to drive down the age of marriage for women.  This results
    # in women being married for more of their child-bearing years,
    # which can drive up total fertility rates, but which also lowers
    # the ages at which women give birth, shortening the time between
    # generations and so driving up the birth rate per unit time, [2].

    # 3. The shortage of Muslim women created by polygamy means some
    # Muslim men will marry non-Muslim women, but still raise their
    # children as Muslims.  This mechanism is missing from all the
    # usual statistics, which deal in fertility rates per woman, and
    # is not modelled here.

    # Modelling the increase in marriage rates.  
    # In [3] it was reported that there were an estimated 20,000
    # polygamous Muslim marriages in the UK in 2014.  In Table 11
    # (p36) of [4] the number of Muslim marriages in the UK in 2011 is
    # given as 335,158, the number of cohabiting couples as 22,665,
    # the number of lone parents as 99,679, and the number of other
    # household types with dependent children as 85,187, for a total
    # of 542,689 households either with children or potentially going
    # to have children.  20,000 polygamous marriages are therefore
    # 3.68% of potential child-bearing Muslim relationships, and this
    # represents a source of child production not available to the
    # parallel non-Muslim society.  Therefore I set

    mtfr = 1.0368 * sr_non_asfr_2011.sum()*5/1000

    # Note that the tendancy for Muslim TFRs to exceed non-Muslim TFRs
    # in the same country was mentioned by the Pew Research Center on
    # p168 in [5].

    # Modelling the lowered birth ages.
    # In this case I've just used the birth age distribution from the
    # six richest Gulf states used in Simulation C.  This distribution
    # reflects what you get in modern developed wealthy Muslim
    # countries practising polygamy.  I scale it here to have the TFR
    # predicted above for Muslims practising polygamy in the UK.  In
    # other words, polygamy can still be expected to drive down birth
    # ages in the UK by creating a 'shortage' of women, but UK women
    # can still choose to have UK-like numbers of children.

    sr_tm = sr_gul_asfr_2009 * mtfr / (sr_gul_asfr_2009.sum()*5/1000)

    # [2] http://www.tandfonline.com/doi/abs/10.1080/00324728.1980.10412838
    # [3] http://www.telegraph.co.uk/culture/tvandradio/11108763/The-Men-with-Many-Wives-the-British-Muslims-who-practise-polygamy.html
    # [4] http://www.mcb.org.uk/wp-content/uploads/2015/02/MCBCensusReport_2015.pdf
    # [5] http://assets.pewresearch.org/wp-content/uploads/sites/11/2011/01/FutureGlobalMuslimPopulation-WebPDF-Feb10.pdf

    # This simulation makes the assumption that Muslim polygamy is not
    # going away, which seems reasonable given that it has been around
    # for 1300 years already.

    df_mus, df_non = runsim(df_mus_2011,df_non_2011,sr_mus_asfr_2011,sr_tm,sr_non_asfr_2011,sr_non_asfr_2011,100,df_uk_dr,pmc,2500)

    sr = 100*df_mus.sum(axis=1)/(df_mus.sum(axis=1)+df_non.sum(axis=1))
    ax1.plot(sr.index,sr.values,label='Projection B')


    # Simulation B : Jihadists

    # In [6] The Times reported that in 2017 the UK was home to 23,000
    # jihadists.  In [7] The Mail reported that the number of Muslims
    # in the UK in 2014 was 3,114,992.  These two figures imply a
    # 'jihadists rate' of 0.0073836 jihadists per Muslim in the UK.

    # It is interesting to note that the 'jihadist rates' in France
    # and Germany are not very different; in March 2017 France had an
    # estimated 17,393, [8], and in August 2016 Germany had an
    # estimated 11,000 or 43,000, depending on your definition, [9].

    # The 'jihadist rate' is a relative frequency and it is therefore
    # reasonable to treat it as a probability, in this case the
    # probability of each individual Muslim being a jihadist.  The
    # number of jihadists to expect in the UK is then this probability
    # multiplied by the Muslim population size.  We can then plot the
    # expected number of jihadists in the UK as the Muslim population
    # rises in the future.  This assumes that the 'jihadist rate'
    # remains constant in the future.

    # The expected number of jihadists is plotted for Simulation B.

    # [6] www.thetimes.co.uk/article/huge-scale-of-terror-threat-revealed-uk-home-to-23-000-jihadists-3zvn58mhq
    # [7] http://www.dailymail.co.uk/news/article-3424584/Muslims-UK-3-million-time-50-born-outside-Britain-Number-country-doubles-decade-immigration-birth-rates-soar.html
    # [8] http://www.senat.fr/rap/r16-483/r16-4837.html
    # [9] https://en.qantara.de/content/germanys-islamist-scene-in-numbers

    # In practice I calculate the jihadist rate using the projected number
    # of Muslims in 2017 rather than the reported number in 2014, so that 
    # my graph correctly displays 23,000 jihadists in 2017.

    jr = 23000/((4*df_mus.sum(axis=1).iloc[1] + df_mus.sum(axis=1).iloc[2])/5)
    ax2 = ax1.twinx()
    ax2.plot(sr.index,jr*df_mus.sum(axis=1).values,
             ls='--',
             c='orange',
             label='Jihadist Projection B')


    # Simulation C : Convergence to Gulf rates.

    # This simulation assumes that the UK Muslim ASFRs will converge
    # linearly over a period of 100 years to match exactly average
    # 2009 ASFRs in the six richest Gulf states, namely Saudi Arabia,
    # Qatar, Oman, Kuwait, Bahrain and the United Arab Emirates.  The
    # rationale for this simulation is that these six Muslim states
    # are developed countries all with GDP (PPP) per capita values
    # greater than the UK's, and as such they offer an alternative
    # prediction of future UK Muslim behaviour to the convergence
    # theories of Simulation A.

    df_mus, df_non = runsim(df_mus_2011,df_non_2011,sr_mus_asfr_2011,sr_gul_asfr_2009,sr_non_asfr_2011,sr_non_asfr_2011,100,df_uk_dr,pmc,2500)

    sr = 100*df_mus.sum(axis=1)/(df_mus.sum(axis=1)+df_non.sum(axis=1))
    ax1.plot(sr.index,sr.values,label='Projection C')

    
    # Pew Research Center Point

    # On p50 of [1] the Pew Research Center projects that the UK population
    # will be 8.3% Muslim in 2050 with no new migration.  I've plotted this
    # datapoint on my graph for comparison with my projections.

    ax1.scatter(2050,8.3,color='red',marker='*',zorder=5,
                label='Pew Research Ctr\nProjection, see\ngoo.gl/6zFh1P p50')

    # The Pew Projection is very close to my own projections for 2050
    # which gives me confidence that I don't have any major bugs in my
    # methods, my data or my code.  The Pew Projection is nevertheless
    # slightly higher than my own Projection A, to which it is most
    # comparable in terms of assumptions made.  This is most likely
    # due to the higher non-Muslim total fertility rate that I use, as
    # discussed in the load_asfrs() function, resulting in a slightly
    # higher non-Muslim population in 2050 in my projection.


    # Put final details on the plot

    ax1.set_xticks(np.arange(2010,2130,10))
    ax1.set_yticks(np.arange(0,22,2))
    ax1.set_ylim(0,21)
    ax1.set_ylabel('% of UK population which is Muslim')
    ax1.set_xlabel('Year')
    ax1.set_xlim(2005,2125)
    ax2.set_ylabel('Number of Jihadists')
    ax2.set_ylim(0,60000)
    ax2.set_yticks(np.arange(0,65000,5000))
    plt.title('Future UK Muslim Population\n',
              fontsize=18,verticalalignment='baseline')
    ax1.legend(loc='upper left',title='Use left axis')
    ax2.legend(loc='upper center',title='Use right axis')

    fig.text(0.355, 0.915, '(with no further immigration)', fontsize=12)

    fig.text(0.1, 0.22, 'Projection A : If UK Muslim fertility rates fall to UK non-Muslim rates over 100 yrs.', fontsize=12)

    fig.text(0.24, 0.198, 'This is the assumption used by Pew, see goo.gl/6zFh1P p185.', fontsize=12)

    fig.text(0.1, 0.16, 'Projection B : Same as Projection A plus an adjustment for Muslim polygamy.', fontsize=12)

    fig.text(0.1, 0.12, 'Projection C : If UK Muslim fertility rates fall to Gulf state rates over 100 yrs.', fontsize=12)

    fig.text(0.12, 0.08, 'All Projections assume zero immigration and zero '\
             'emigration of all peoples.', fontsize=12)

    fig.text(0.17, 0.058, 'Jihadist Projection B uses 2017 rate of 23,000 per 3.2m Muslims.', fontsize=12)

    fig.text(0.21, 0.02, 'Fully annotated and referenced code available from goo.gl/L3XKc8', fontsize=10)

    fig.text(0.19, 0.005, 'All Rights Released; this work is in the Public Domain; produced in 2017.', fontsize=10)

