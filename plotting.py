import coffea
from coffea import hist, processor
import coffea.hist as hist
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import optimize
import numpy as np

etabins=np.array([-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964,
         -2.853, -2.65, -2.5, -2.322, -2.172, -2.043, -1.93, -1.83, -1.74, -1.653, -1.566, -1.479, -1.392, -1.305,
         -1.218, -1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174,
         -0.087, 0, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.879, 0.957, 1.044, 1.131,
         1.218, 1.305, 1.392, 1.479,1.566, 1.653, 1.74, 1.83, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853, 2.964,
         3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191])
nEta = 82
etaC=0.5*(etabins[:-1]+etabins[1:])

def plotStack(output, x_var, y_var, norm_var, x_low, x_high, ylabel, flavor, ymax, ratioM):
    
    mu_range = slice(x_low, x_high)
    mu_av = 0.5*(mu_range.stop+mu_range.start)

    offset_vs_eta_data = output[y_var].integrate('dataset','Data').integrate("nPU",mu_range)
    offset_vs_eta_mc = output[y_var].integrate('dataset','MC').integrate("nPU",mu_range)
    norm_data = output[norm_var].integrate("dataset","Data").integrate("nPU",mu_range).values()[()] * mu_av 
    norm_mc = output[norm_var].integrate("dataset","MC").integrate("nPU",mu_range).values()[()] * mu_av
    
    offset_vs_eta_data.scale(1/norm_data)
    offset_vs_eta_mc.scale(1/norm_mc)
    keys = list(offset_vs_eta_data.values().keys())
    
    offset_v_eta_data = offset_vs_eta_data#.remove(["lep","untrk"],"flavor")
    offset_v_eta_mc = offset_vs_eta_mc#.remove(["lep","untrk"],"flavor")

    bin_l = etabins[:16]
    bin_c = etabins[11:-11]
    bin_r = etabins[-16:]
    
    m_chm = offset_v_eta_mc.values()['chm',][11:-11]
    m_chu = offset_v_eta_mc.values()['chu',][11:-11]
    m_nh = offset_v_eta_mc.values()['nh',][11:-11]
    m_ne = offset_v_eta_mc.values()['ne',][11:-11]
    m_hfe_l = offset_v_eta_mc.values()['hfe',][:15]
    m_hfh_l = offset_v_eta_mc.values()['hfh',][:15]
    m_hfe_r = offset_v_eta_mc.values()['hfe',][-15:]
    m_hfh_r = offset_v_eta_mc.values()['hfh',][-15:]

    d_chm = offset_v_eta_data.values()['chm',][11:-11]
    d_chu = offset_v_eta_data.values()['chu',][11:-11]
    d_nh = offset_v_eta_data.values()['nh',][11:-11]
    d_ne = offset_v_eta_data.values()['ne',][11:-11]
    d_hfe_l = offset_v_eta_data.values()['hfe',][:15]
    d_hfh_l = offset_v_eta_data.values()['hfh',][:15]
    d_hfe_r = offset_v_eta_data.values()['hfe',][-15:]
    d_hfh_r = offset_v_eta_data.values()['hfh',][-15:]

    r_pfchs = np.where((offset_vs_eta_mc.remove(["lep","untrk", "chm"],"flavor")).integrate("flavor").values()[()]>0,
        ((offset_vs_eta_data.remove(["lep","untrk", "chm"],"flavor")).integrate("flavor").values()[()]/(offset_vs_eta_mc.remove(["lep","untrk", "chm"],"flavor")).integrate("flavor").values()[()]), 0)

    r_chm = np.where(m_chm>1e-5, d_chm/m_chm, 0)
    r_chu = np.where(m_chu>1e-5, d_chu/m_chu, 0)
    r_nh = np.where(m_nh>1e-5, d_nh/m_nh, 0)
    r_ne = np.where(m_ne>1e-5, d_ne/m_ne, 0)
    r_hfh_l = np.where(m_hfh_l>1e-5, d_hfh_l/m_hfh_l, 0)
    r_hfe_l = np.where(m_hfe_l>1e-5, d_hfe_l/m_hfe_l, 0)
    r_hfh_r = np.where(m_hfh_r>1e-5, d_hfh_r/m_hfh_r, 0)
    r_hfe_r = np.where(m_hfe_r>1e-5, d_hfe_r/m_hfe_r, 0)
    
    fig = plt.figure(figsize=(6, 6)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
    ax = plt.subplot(gs[0])
    rax = plt.subplot(gs[1])
    fig.subplots_adjust(hspace=.07)
    
    ax.set_ylim(0,ymax)
    ax.set_xlim(-5.2, 5.2)
    ax.set_xticklabels([])
    ax.set_ylabel(ylabel)
    rax.grid()
    
    if flavor == 'all':
    
        hep.histplot([m_ne, m_nh, m_chu, m_chm], bin_c, stack=True, histtype='fill',
                     ax=ax, color=['blue', 'lime', 'salmon', 'red'],
                     label=['Photons','Neutral Hadrons', 'Unssoc. Charged Hadrons','Assoc. Charged Hadrons' ])
        hep.histplot([m_hfe_l, m_hfh_l], bin_l, stack=True, histtype='fill',
                     ax=ax, color=['darkviolet', 'fuchsia'],
                     label=['EM Deposits', 'Hadron Deposits'])
        hep.histplot([m_hfe_r, m_hfh_r], bin_r, stack=True, histtype='fill',
                     ax=ax, color=['darkviolet', 'fuchsia'],
                    label=['EM Deposits', 'Hadron Deposits'])

        hep.histplot([d_ne, d_nh, d_chu, d_chm], bin_c, stack=True, histtype='errorbar',
                     marker=["x", "d", "o", "o"], ax=ax,
                     label=['Photons','Neutral Hadrons', 'Unssoc. Charged Hadrons','Assoc. Charged Hadrons' ],
                     fillstyle='none', color='black', markersize=5,)
        hep.histplot([d_hfe_l, d_hfh_l], bin_l, stack=True, histtype='errorbar',
                     marker=["*", "^"], ax=ax, label=['EM Deposits', 'Hadron Deposits'],
                     fillstyle='none', color='black', markersize=5,)
        hep.histplot([d_hfe_r, d_hfh_r], bin_r, stack=True, histtype='errorbar',
                     marker=["*", "^"], ax=ax, label=['EM Deposits', 'Hadron Deposits'],
                     fillstyle='none', color='black', markersize=5,)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[i], handles[8+i][0]) for i in range(6)],
                 ['Assoc. Charged Hadrons', 'Unssoc. Charged Hadrons', 'Neutral Hadrons', 'Photons',
                  'Hadron Deposits', 'EM Deposits'], title='Markers: Data, Histograms: MC')

        if (ratioM):
            hep.histplot([r_ne, r_nh, r_chu, r_chm], bin_c, stack=False, histtype='errorbar',
                         marker=["x", "d", "o", "o"], ax=rax,
                         fillstyle='none', color=['blue', 'lime', 'salmon', 'red'], markersize=5,)
            hep.histplot([r_hfe_l, r_hfh_l], bin_l, stack=False, histtype='errorbar',
                         marker=["*", "^"], ax=rax,
                         fillstyle='none', color=['darkviolet', 'fuchsia'], markersize=5,)
            hep.histplot([r_hfe_r, r_hfh_r], bin_r, stack=False, histtype='errorbar',
                         marker=["*", "^"], ax=rax,
                         fillstyle='none', color=['darkviolet', 'fuchsia'], markersize=5,)

        else:
            hep.histplot(r_pfchs, etabins, stack=False, histtype='errorbar',
                        marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
            rax.legend()
        
    elif flavor == 'chm':
        hep.histplot(m_chm, bin_c, stack=False, histtype='fill', 
                     ax=ax, color='red', label=['Assoc. Charged Hadrons'] )
        hep.histplot(d_chm, bin_c, stack=False, histtype='errorbar', marker="o", ax=ax,
                     label=['Assoc. Charged Hadrons' ], fillstyle='none', color='black', markersize=5,)
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[0], handles[1][0])],
                  ['Assoc. Charged Hadrons'], title='Markers: Data, Histograms: MC')
        hep.histplot(r_chm, bin_c, stack=False, histtype='errorbar',
                     marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
    elif flavor == 'chu':
        hep.histplot(m_chu, bin_c, stack=False, histtype='fill', 
                     ax=ax, color='salmon', label=['Unassoc. Charged Hadrons'] )
        hep.histplot(d_chu, bin_c, stack=False, histtype='errorbar', marker="o", ax=ax,
                     label=['Unassoc. Charged Hadrons' ], fillstyle='none', color='black', markersize=5,)
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[0], handles[1][0])],
                  ['Unassoc. Charged Hadrons'], title='Markers: Data, Histograms: MC')
        hep.histplot(r_chu, bin_c, stack=False, histtype='errorbar',
                     marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
    elif flavor == 'nh':
        hep.histplot(m_nh, bin_c, stack=False, histtype='fill', 
                     ax=ax, color='lime', label=['Neutral Hadrons'] )
        hep.histplot(d_nh, bin_c, stack=False, histtype='errorbar', marker="d", ax=ax,
                     label=['Neutral Hadrons' ], fillstyle='none', color='black', markersize=5,)
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[0], handles[1][0])],
                  ['Neutral Hadrons'], title='Markers: Data, Histograms: MC')
        hep.histplot(r_nh, bin_c, stack=False, histtype='errorbar',
                     marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
    elif flavor == 'ne':
        hep.histplot(m_ne, bin_c, stack=False, histtype='fill', 
                     ax=ax, color='blue', label=['Photons'] )
        hep.histplot(d_ne, bin_c, stack=False, histtype='errorbar', marker="x", ax=ax,
                     label=['Photons' ], fillstyle='none', color='black', markersize=5,)
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[0], handles[1][0])],
                  ['Photons'], title='Markers: Data, Histograms: MC')
        hep.histplot(r_ne, bin_c, stack=False, histtype='errorbar',
                     marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
        
    elif flavor == 'hfh':
    
        hep.histplot(m_hfh_l, bin_l, stack=False, histtype='fill',
                     ax=ax, color='fuchsia', label='Hadron Deposits')
        hep.histplot(m_hfh_r, bin_r, stack=False, histtype='fill',
                     ax=ax, color='fuchsia', label='Hadron Deposits')
        hep.histplot(d_hfh_l, bin_l, stack=False, histtype='errorbar',
                     marker="^", ax=ax, label='Hadron Deposits',
                     fillstyle='none', color='black', markersize=5,)
        hep.histplot(d_hfh_r, bin_r, stack=False, histtype='errorbar',
                     marker="^", ax=ax, label='Hadron Deposits',
                     fillstyle='none', color='black', markersize=5,)

        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[0], handles[2][0])],
                  ['Hadron Deposits'], title='Markers: Data, Histograms: MC')
        hep.histplot(r_hfh_l, bin_l, stack=False, histtype='errorbar',
                     marker="o", ax=rax,
                     fillstyle='none', color='black', markersize=5,)
        hep.histplot(r_hfh_r, bin_r, stack=False, histtype='errorbar',
                     marker="o", ax=rax,
                     fillstyle='none', color='black', markersize=5,)
    elif flavor == 'hfe':
    
        hep.histplot(m_hfe_l, bin_l, stack=False, histtype='fill',
                     ax=ax, color='darkviolet',
                     label='EM Deposits')
        hep.histplot(m_hfe_r, bin_r, stack=False, histtype='fill',
                     ax=ax, color='darkviolet',
                    label='EM Deposits')
        hep.histplot(d_hfe_l, bin_l, stack=False, histtype='errorbar',
                     marker="*", ax=ax, label='EM Deposits',
                     fillstyle='none', color='black', markersize=5,)
        hep.histplot(d_hfe_r, bin_r, stack=False, histtype='errorbar',
                     marker="*", ax=ax, label='EM Deposits',
                     fillstyle='none', color='black', markersize=5,)

        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([(handles[0], handles[2][0])],
                  ['EM Deposits'], title='Markers: Data, Histograms: MC')
        hep.histplot(r_hfe_l, bin_l, stack=False, histtype='errorbar',
                     marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
        hep.histplot(r_hfe_r, bin_r, stack=False, histtype='errorbar',
                     marker="o", ax=rax, fillstyle='none', color='black', markersize=5,)
    rax.set_ylim(0.4,1.8)
    rax.set_xlim(-5.2, 5.2)
    rax.set_xlabel(r'$\eta$')
    rax.set_ylabel('Data/MC')
    plt.show()

def plotHist(output, variable, xlabel):
    scales = output[variable].integrate(variable).values()
    data_scale = scales['Data',]
    mc_scale = scales['MC',]
    scales['Data'] = 1/data_scale
    scales['MC'] = 1/mc_scale
    del scales['Data',]
    del scales['MC',]
    output[variable].scale(scales, axis='dataset')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax = hist.plot1d(output[variable],overlay='dataset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    scales['Data'] = data_scale
    scales['MC'] = mc_scale
    output[variable].scale(scales, axis='dataset')

def plotProfile(output, x_var, hist_2d, xlow, xhigh, xlabel, ylabel):
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(5,5),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)
    num1 = output[hist_2d].integrate('dataset','Data') 
    denom1 = output[x_var].integrate('dataset','Data') 
    hist.plotratio(num1, denom1, unc='num', ax=ax,
                   error_opts={'marker':'o', 'markersize': 3.,
                               'markeredgecolor':'k', 'color':'k',
                               'elinewidth': 0.5, },label='Data', clear=False,)
    num2 = output[hist_2d].integrate('dataset','MC') 
    denom2 = output[x_var].integrate('dataset','MC') 
    hist.plotratio(num2, denom2, unc='num',ax=ax,
                   error_opts={'marker':'o', 'markersize': 3.,
                               'markeredgecolor':'r', 'color':'none',
                               'elinewidth': 1.0, },label='MC',clear=False)
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(0, 60)
    rax.set_ylim(0.7, 1.3)
    ax.set_xlabel(None)
    rax.set_xlabel(xlabel)
    rax.set_ylabel('Data/MC')
    ax.set_ylabel(ylabel)
    ax.grid(color='b', ls = '-.', lw = 0.25)
    ax.legend()
    nbins = len(output[x_var].values()['Data',])
    bin_center = 0.5*(np.linspace(0,nbins/2,nbins+1)[:-1]+np.linspace(0,nbins/2,nbins+1)[1:])
    im = rax.scatter(bin_center,(num1.values()[()]*denom2.values()[()])/(num2.values()[()]*denom1.values()[()]),
                     s=5, marker='o', facecolors='none', edgecolor='k')
    im = rax.plot(bin_center, np.ones(len(bin_center)), '--', color='gray')

def func(x, a, b, c):
    return (a * x**2) + (b*x) + c

def fitProfile(output, x_var, y_var, norm_var, hist_2d, xlabel, ylabel, nbin, xmin, xmax):
    '''Need to add text in the plot from fitting and display eta range.'''
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(5,5),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)
    
    # print(etabins[nbin],'-',etabins[nbin+1])
    eta_range = slice(etabins[nbin], etabins[nbin+1])
    offset_vs_eta_data = output[y_var].integrate('dataset','Data').integrate('eta',eta_range )
    norm_data = ((output[norm_var].integrate('dataset','Data')).integrate('eta',eta_range ).integrate('flavor','chm')).values()[()]
    offset_v_eta_data = (offset_vs_eta_data.remove(["lep","untrk", 'chm'],"flavor")).integrate('flavor')

    offset_vs_eta_mc = output[y_var].integrate('dataset','MC').integrate('eta',eta_range )
    norm_mc = (((output[norm_var].integrate('dataset','MC')).integrate('eta',eta_range ).integrate('flavor','chm')).values()[()])
    offset_v_eta_mc = (offset_vs_eta_mc.remove(["lep","untrk", 'chm'],"flavor")).integrate('flavor')
    
    '''converting mu to rho for x axis values'''
    num1 = (output['p_rho_nPU']).integrate('dataset','Data') 
    denom1 = (output['nPU']).integrate('dataset','Data') 
    x_data = np.where(denom1.values()[()]>0., num1.values()[()]/denom1.values()[()], 0)
    
    num2 = (output['p_rho_nPU']).integrate('dataset','MC') 
    denom2 = (output['nPU']).integrate('dataset','MC') 
    x_mc = np.where(denom2.values()[()]>0., num2.values()[()]/denom2.values()[()], 0)
    
    nbins = len(output[x_var].values()['Data',])
    bin_center = (0.5*(np.linspace(0,nbins/2,nbins+1)[:-1]+np.linspace(0,nbins/2,nbins+1)[1:])).astype(int)
    yvals_d = (offset_v_eta_data.values()[()]* (1/norm_data))
    yvals_m = (offset_v_eta_mc.values()[()]* (1/norm_mc))
    yerr_d = (offset_v_eta_data.values(sumw2=True)[()])[1]* (1/norm_data)
    yerr_m = (offset_v_eta_mc.values(sumw2=True))[()][1]* (1/norm_mc)
    im = ax.scatter(x_data, yvals_d, label='Data', s=15, facecolors='k', edgecolors='k')
    im = ax.scatter(x_mc, yvals_m, label='MC', s=15, facecolors='none', edgecolors='r')
    
    min_xlim=np.min(x_data[2*xmin:2*xmax+1]) if np.min(x_data[2*xmin:2*xmax+1])<np.min(x_mc[2*xmin:2*xmax+1]) else np.min(x_mc[2*xmin:2*xmax+1])
    max_xlim=np.max(x_data[2*xmin:2*xmax+1]) if np.max(x_data[2*xmin:2*xmax+1])>np.max(x_mc[2*xmin:2*xmax+1]) else np.max(x_mc[2*xmin:2*xmax+1])
    ax.set_xlim(min_xlim, max_xlim)
    
    ymax1 = np.max(yvals_d[int(2*xmin):int(2*xmax)+1])
    ymax2 = np.max(yvals_m[int(2*xmin):int(2*xmax)+1])
    ymax = ymax1 if(ymax1>ymax2) else ymax2
    ax.set_ylim(0, 1.2*ymax)
    ax.set_ylabel(ylabel)

    im = rax.scatter(x_data, yvals_d/yvals_m,
                     facecolors='none', edgecolors='k', s=15)
    im = rax.plot(bin_center, np.ones(len(bin_center)), '--', color='k')
    rax.set_ylim(0.5, 1.6)
    rax.set_xlabel(xlabel)
    rax.set_ylabel('Data/MC')
    
    
    d_params= optimize.curve_fit(func, x_data[int(2*xmin):int(2*xmax)+1],
                                                   yvals_d[int(2*xmin):int(2*xmax)+1],
                                                   p0=[2, 2, 2], full_output=True)
    m_params= optimize.curve_fit(func, x_mc[int(2*xmin):int(2*xmax)+1],
                                                       yvals_m[int(2*xmin):int(2*xmax)+1],
                                                       p0=[2, 2, 2], full_output=True)
    d_err = np.sqrt(np.diag(d_params[1]))
    m_err = np.sqrt(np.diag(m_params[1]))
    # print(d_err, m_err)

    #print(yerr_d[int(2*xmin):int(2*xmax)+1])
    #print(yerr_m[int(2*xmin):int(2*xmax)+1])
    
    chisq_d = np.sum(((yvals_d[int(2*xmin):int(2*xmax)+1] -
                    func( bin_center[int(2*xmin):int(2*xmax)+1], d_params[0][0], d_params[0][1], d_params[0][2]))
                     /yerr_d[int(2*xmin):int(2*xmax)+1])**2)
    chisq_m = np.sum(((yvals_m[int(2*xmin):int(2*xmax)+1] -
                    func( bin_center[int(2*xmin):int(2*xmax)+1], m_params[0][0], m_params[0][1], m_params[0][2]))
                     /yerr_m[int(2*xmin):int(2*xmax)+1])**2)
    # print (chisq_d, chisq_m)

    ax.plot(bin_center[int(xmin):int(2*xmax)+1], func( bin_center[int(xmin):int(2*xmax)+1], d_params[0][0], d_params[0][1], d_params[0][2]), 'k')
    ax.plot(bin_center[int(xmin):int(2*xmax)+1], func( bin_center[int(xmin):int(2*xmax)+1], m_params[0][0], m_params[0][1], m_params[0][2]), 'r')
    
    ax.text(min_xlim, 1.21*ymax, 'CMS', {'color': 'k', 'fontsize': 12, 'fontweight':'black', 'fontfamily':'sans'}, va="bottom", ha="left")
    ax.text(max_xlim, 1.21*ymax, r'AK4 PFchs {} $\leq$ $\eta$ $\leq$ {}'.format(etabins[nbin],etabins[nbin+1]), {'color': 'k', 'fontsize': 10}, va="bottom", ha="right")
    ax.text(1.3*min_xlim, 1.05*ymax, 'Data', {'color': 'k', 'fontsize': 10}, va="top", ha="left", weight='bold')
    #ax.text(1.3*min_xlim, 1.1*ymax, 'Data', {'color': 'k', 'fontsize': 10}, va="top", ha="left", weight='bold')
    #ax.text(1.3*min_xlim, 1.05*ymax, r'$\chi^2/ndof = {:1.3f}/{}$ '.format(12.3456,len(yvals_d[int(xmin):int(xmax)])-1), {'color': 'k', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.98*ymax, r'p0 = {:1.3f} $\pm$ {:1.3f}'.format(d_params[0][2], d_err[2]), {'color': 'k', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.93*ymax, r'p1 = {:1.3f} $\pm$ {:1.3f}'.format(d_params[0][1], d_err[1]), {'color': 'k', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.88*ymax, r'p2 = {:1.3f} $\pm$ {:1.3f}'.format(d_params[0][0], d_err[0]), {'color': 'k', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.77*ymax, 'MC', {'color': 'r', 'fontsize': 10}, va="top", ha="left", weight='bold')
    #ax.text(1.3*min_xlim, 0.82*ymax, 'MC', {'color': 'r', 'fontsize': 10}, va="top", ha="left", weight='bold')
    #ax.text(1.3*min_xlim, 0.77*ymax, r'$\chi^2/ndof = {:1.3f}/{}$ '.format(12.3456,len(yvals_m[int(xmin):int(xmax)])-1), {'color': 'r', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.7*ymax, r'p0 = {:1.3f} $\pm$ {:1.3f}'.format(m_params[0][2], m_err[2]), {'color': 'r', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.65*ymax, r'p1 = {:1.3f} $\pm$ {:1.3f}'.format(m_params[0][1], m_err[1]), {'color': 'r', 'fontsize': 9}, va="top", ha="left")
    ax.text(1.3*min_xlim, 0.6*ymax, r'p2 = {:1.3f} $\pm$ {:1.3f}'.format(m_params[0][0], m_err[0]), {'color': 'r', 'fontsize': 9}, va="top", ha="left")