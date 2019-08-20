import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pymongo import MongoClient
from datetime import datetime
import warnings  
warnings.filterwarnings('ignore')
from datetime import date
import calendar
from random import sample
# Jupyter Notebook
import plotly.express as px
import chart_studio.plotly as py
from plotly.offline import iplot,init_notebook_mode
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import random

def plotBox(df,listcol,prev_df=None):
    if prev_df is None:
        data = []
        colors = colorSanity(len(listcol))
        for i,n in enumerate(listcol):
            trace = go.Violin(
                y=df[n],
                box_visible=True,
                meanline_visible=True,
                name = n.split('_')[-1],
                
                opacity = 0.6,
                marker = dict(
                    color = colors[i]
                )
            )
            data.append(trace)
        iplot(data)
    else:
        data = []
        colors = colorSanity(len(listcol))
        for i,n in enumerate(listcol):
            trace = go.Violin(
                y=prev_df[n],
                box_visible=False,
                meanline_visible=True,
                legendgroup='Prev Week', scalegroup='Prev Week',
                name = n.split('_')[-1],
                opacity=0.6,
                side='negative',
                marker = dict(
                    color = 'orange'
                )
            )
            data.append(trace)
            trace = go.Violin(
                y=df[n],
                box_visible=False,
                meanline_visible=True,
                name = n.split('_')[-1],
                opacity=0.6,
                legendgroup='This Week', scalegroup='This Week',
                side='positive',
                marker = dict(
                    color = 'blue'
                )
            )
            data.append(trace)

        iplot(data)
        
def extractCol(df,colname):
    def oneHot(x,name):
        try:
            return x[name]
        except TypeError:
            return 0
    __list = list(df[colname][random.randint(0, len(df))].keys())
    __list.sort()
    for __name in __list:
        df[colname+'_'+__name] = df[colname].apply(lambda x : oneHot(x,__name))
        
    return df

def colorSanity(n):
    try:
        return px.colors.qualitative.Dark24[0:n] 
    except:
        return ['blue','orange']
    

def plotViolinPrev(df,prev_df,list_col,name='',color=['orange','blue']):
    fig = go.Figure()
    color = colorSanity(len(list_col))
    for i,n in enumerate(list_col):
        fig.add_trace(go.Violin(
                            y=prev_df[n],
                            legendgroup=n, scalegroup=n, 
                            name="previous "+n.split('_')[-1],
                            line_color=color[i],opacity=0.4)
        )
        fig.add_trace(go.Violin(
                            y=df[n],
                            legendgroup=n, scalegroup=n, 
                            name=n.split('_')[-1],
                            line_color=color[i],opacity=1)
                )

    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0,violinmode='overlay')
    fig.show()

def plotSpline(dataframe,listcol,name='',x='Time',y='Value'):
    df = dataframe.set_index('date').groupby(pd.Grouper(freq='H')).sum()
    colors = colorSanity(len(listcol))
    data = []
    for i,n in enumerate(listcol):
        trace = go.Scatter(
                        x = df.index,
                        y = df[n],
                        line_shape='spline',
                        name = n.split('_')[-1],
                        
                        marker = dict(color = colors[i]))
        data.append(trace)
    layout = dict(title = name,
                  xaxis= dict(title= x,ticklen= 5,zeroline= True),
                  yaxis= dict(title= y,ticklen= 5,zeroline= True)
                 )
    fig = dict(data = data, layout = layout)
    iplot(fig)


def plotPiePrev(df,prev_df,listcol):
    tmp = df.sum()[listcol]
    pie1_list = tmp.values

    tmp = prev_df.sum()[listcol]
    pie2_list = tmp.values

    labels = tmp.index.map(lambda x : x.split('_')[-1])


    fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['Previous week', 'This week'])


    fig.add_trace(go.Pie(labels=labels,
                         values=pie2_list, 
                         scalegroup='one',
                         opacity=0.7,
                         name="Previous week"), 1, 1)
    fig.add_trace(go.Pie(labels=labels,
                         values=pie1_list, 
                         scalegroup='one',
                         name="This week"), 1, 2)

    iplot(fig)


def plotCompareDay(df,listcol,name='',x='Time',y='Value'):
    tmp = df.set_index('date').groupby(pd.Grouper(freq='H')).sum()
    tmp['day'] = tmp.index.day
    tmp['hour'] = tmp.index.hour
    data = []
    palette = [px.colors.sequential.Blues, px.colors.sequential.BuGn, px.colors.sequential.BuPu, px.colors.sequential.Greys, 
    px.colors.sequential.OrRd, px.colors.sequential.PuRd, px.colors.sequential.PuBuGn]
    cnt = 0
    for g in tmp['day'].unique():
        color = palette[cnt]
        cnt+=1
        for i,n in enumerate(listcol):
            trace = go.Scatter(
                            x = tmp.hour,
                            y = tmp[n][tmp['day']==g],
                            line_shape='spline',
                            legendgroup=str(g),
                            name = 'Day '+str(g)+' '+n.split('_')[-1],
                            marker = dict(color = color[(-i)-1]))
            data.append(trace)
    layout = dict(title = name,
                  xaxis= dict(title= x,ticklen= 5,zeroline= True),
                  yaxis= dict(title= y,ticklen= 5,zeroline= True),
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)'
                 )
    fig = dict(data = data, layout = layout)
    iplot(fig)


def plotBar(tmp,prev_tmp,listcol,name=''):
    data=[]
    colors = colorSanity(len(listcol))
    for i,n in enumerate(listcol):
        trace = go.Bar(
                        x = prev_tmp.index,
                        y = prev_tmp[n],
                        legendgroup=n,
                        name = "previous "+n.split('_')[-1],
                        marker = dict(color = colors[i],
                                     line=dict(color='rgb(0,0,0)',width=0.5)),
                        opacity=0.6,
                        text = prev_tmp.index)
        data.append(trace)
        trace = go.Scatter(
                        x = tmp.index,
                        y = tmp[n],
                        legendgroup=n,
                        name = n.split('_')[-1],
                        mode='lines+markers',
                        marker = dict(color = colors[i],
                                     line=dict(color='red',width=0.5)),
                        opacity=0.8,
                        text = tmp.index)
        data.append(trace)

    layout = go.Layout(barmode = "group",title=name)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)


def plotOneBar(tmp,prev_tmp,listcol,name='',colors='blue'):
    data=[]
    n = listcol[0]
    trace = go.Bar(
                    x = tmp.index,
                    y = tmp[n],
                    name = n.split('_')[-1],
                    marker = dict(color = colors,
                                    line=dict(color='rgb(0,0,0)',width=0.5)),
                    opacity=0.8,
                    text = tmp.index)
    data.append(trace)
    trace = go.Scatter(
                    x = prev_tmp.index,
                    y = prev_tmp[n],
                    name = "previous "+n.split('_')[-1],
                    mode='lines',
                    marker = dict(color = 'red',
                                    line=dict(color='red',width=0.5)),
                    opacity=1,
                    text = prev_tmp.index)
    data.append(trace)

    layout = go.Layout(title=name)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)


def plotRateBar(tmp,prev_tmp,listcol,name=''):
    data=[]
    colors = colorSanity(len(listcol))
    for i,n in enumerate(listcol):
        trace = go.Bar(
                        x = tmp.index,
                        y = tmp[n]/tmp[n].sum(),
                        name = n.split('_')[-1],
                        legendgroup = n,
                        marker = dict(color = colors[i],
                                     line=dict(color='rgb(0,0,0)',width=0.5)),
                        text = tmp.index)
        data.append(trace)
        trace = go.Scatter(
                        x = prev_tmp.index,
                        y = prev_tmp[n]/prev_tmp[n].sum(),
                        name = "previous "+n.split('_')[-1],
                        legendgroup = n,
                        mode='lines+markers',
                        marker = dict(color = colors[i],
                                     line=dict(color='red',width=0.5)),
                        text = prev_tmp.index)
        data.append(trace)
    

    layout = go.Layout(barmode = "group",title=name)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    
def plotOneRateBar(tmp,prev_tmp,listcol,name='',colors='blue'):
    data=[]
    n = listcol[0]
    trace = go.Bar(
                    x = tmp.index,
                    y = tmp[n]/tmp[n].sum(),
                    name = n.split('_')[-1],
                    opacity=0.8,
                    marker = dict(color = colors,
                                 line=dict(color='rgb(0,0,0)',width=0.5)),
                    text = tmp.index)
    data.append(trace)
    trace = go.Scatter(
                    x = prev_tmp.index,
                    y = prev_tmp[n]/prev_tmp[n].sum(),
                    name = "previous "+n.split('_')[-1],
                    mode='lines',
                    opacity=1,
                    marker = dict(color = 'red',
                                 line=dict(color='red',width=0.5)),
                    text = prev_tmp.index)
    data.append(trace)
    

    layout = go.Layout(barmode = "group",title=name)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)


def plotCombineBar(tmp,listcol,prev_tmp,prevcol,name='',xaxis='X',yaxis='Y'):
    data=[]
    for i,n in enumerate(listcol):
        trace = {
          'x': tmp.index,
          'y': tmp[n],
          'name': n.split('_')[-1],
          'type': 'bar'
        }
        data.append(trace)
    trace = {
        'x': prev_tmp.index,
        'y': prev_tmp[prevcol],
        'name': 'previous '+prevcol.split('_')[-1],
        'type': 'scatter'
    }
    data.append(trace)
    layout = {
        'xaxis': {'title': xaxis},
        'yaxis': {'title': yaxis},
        'barmode': 'relative',
        'title': name
    };
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    
def plotChangeRate(tmp,prev_tmp,col,x='name',color_continuous_scale = px.colors.diverging.RdYlGn, title = 'Change Rate'):
    tmp['change_rate'] = (tmp[col] - prev_tmp[col]) / prev_tmp[col]
    tmp = tmp.sort_values(['change_rate'])

    fig = px.bar(tmp.reset_index(), x=x, y='change_rate',
                 color='change_rate',
                 color_continuous_midpoint=0,
                 color_continuous_scale = color_continuous_scale,
                title=title)
    fig.show()