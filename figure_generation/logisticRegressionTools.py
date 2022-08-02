import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl

def mcfadden(objs, criteria, predictors, predictor_names):
    """
    Loops through the provided predictors to determine the best McFadden pseudo-R^2
    value.

    Parameters
    ----------
    objs : pandas dataframe
        Dataframe containing performance objective/robustness values.
    criteria : a number (float or int)
        The criterion for the objective to be considered to a 'success'.
    predictors : numpy array
        A size-2 array containing the criteria's two most influential parameters.
    predictor_names : string array
        A size-2 array of strings containing the names of the criteria's two 
        most influenctial parameters.

    Returns
    -------
    mcfaddens : pandas dataframe
        A size (1x2) dataframe containing the McFadden pseudo-R^2 value for each predictor.

    """
	# loop through DU factors to determine the best McFadden psuedo R^2
    objs['Success'] = (criteria).astype(int)
    mcfaddens = np.zeros([np.shape(predictor_names)[0],1])
    for i in range(0,np.shape(predictor_names)[0]):
        logit = sm.Logit(objs['Success'], predictors[['intercept', predictor_names[i]]])
        result = logit.fit()
        mcfaddens[i] = (result.prsquared).astype(float)

    mcfaddens = pd.DataFrame(np.transpose(mcfaddens), columns=predictor_names)
    return mcfaddens

def fitAllLogit(objs, criteria, LHsamples):
    """
    Fits a logistic regression on the input objective/robustness values given its
    two most influential criteria.
    
    Made for DV DU combo, only needs LHsamples dfs, not columns

    Parameters
    ----------
    objs : pandas dataframe
        A dataframe contining the robustness or performance objective values
    criteria : pandas dataframe
        A dataframe contining information on whether a value is deemed a success or failure.
    LHsamples : pandas dataframe
        A (N_RDMSx3) dataframe containing values of the LR intercept and the values of the
        two most important parameters.

    Returns
    -------
    result : logistic regression object
        The result of performing logistic regreession.

    """

    objs['Success'] = (criteria).astype(int)

    logit = sm.Logit(objs['Success'], LHsamples)
    result = logit.fit()

    return result

# plot Logistic regression contours, also adapted from Julie
def plotCombinedFactorMaps_SOS(ax, result, criteria_name, xlabel, ylabel, 
                               x_lim1, x_lim2, y_lim1, y_lim2,
                               dv1_og, dv2_og, dv1_w, dv2_w, 
                               DVconstPredictors, DVconstBaseValues, levels):  
    """
    Plots the logistic regression contour. Adapted from Julie Quinn.

    Parameters
    ----------
    ax : matplotlib object
        The subplot to generate the LR contours for.
    result : logistic regssion object
        The result of performing logistic regreession.
    criteria_name : string
        DESCRIPTION.
    xlabel : string
        x-axis label for ax.
    ylabel : string
        y-axis label for ax.
    x_lim1 : float
        x-axis lower bound.
    x_lim2 : float
        x-axis upper bound.
    y_lim1 : float
        y-axis lower bound..
    y_lim2 : float
        y-axis upper bound.
    dv1_og : float
        Original x-parameter value.
    dv2_og : float
        Original y-parameter value.
    dv1_w : float
        Perturbed x-parameter value.
    dv2_w : TYPE
        Perturbed y-parameter value.
    DVconstPredictors : list
        x-values to be held constant.
    DVconstBaseValues : list
        y-values to be held constant.
    levels : numpy array
        Contour resolution.

    Returns
    -------
    None.

    """
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(criteria_name, fontsize=14)

    
	# set up grid for plotting
    xgrid = np.arange(x_lim1,x_lim2,0.01)
    ygrid = np.arange(y_lim1,y_lim2,0.01)

	# prepare contour data
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    grid = np.column_stack([np.ones(len(x)),x,y])
    
    if len(DVconstPredictors) > 0:
        for i, constant in enumerate(DVconstPredictors):
            grid = np.column_stack([grid, np.ones(len(x))*DVconstBaseValues.iloc[0][constant]])
    
    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))
    
    ax.contourf(X, Y, Z, levels, cmap='RdBu')
    ax.scatter(dv1_og, dv2_og, marker='^', s=160, c='k', label='Baseline')
    ax.scatter(dv1_w, dv2_w, marker='X', s=160, c='k', 
               label='Worst robustness')

