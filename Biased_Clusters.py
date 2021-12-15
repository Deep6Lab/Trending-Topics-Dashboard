import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score

def get_clusters_dist(predictions):
    '''
    Create a dataframe of cluster distribution per year based on the number of articles belonging to the clusters.
    
    Parameter:
    - predictions: predictions dataframe
    '''

    # get cluster distribution per year
    clusters_dist = pd.DataFrame(predictions.groupby(['Year', 'Topic Id'])['Topic Id'].count()).rename(
                                           columns={'Topic Id': 'Number of Articles'}).reset_index()
    
    return clusters_dist


# get top terms
def get_top_keywords(predicted, n_terms, terms_matrix_df, terms_label):
    
    terms = []
    keywords = terms_matrix_df.groupby(predicted).mean()
    for i,r in keywords.iterrows():
        terms.append(', '.join([terms_label[t] for t in np.argsort(r)[-n_terms:]]))
    return terms


def get_silhouette(df, x, m:int=1, n_clusters=15):
    # create a dictionary of step: {'step': m}
    result = dict(step=m)
    
    x2 = np.zeros((x.shape[0],x.shape[1]+1))
    # apply biased amount to Year_Scaled
    x2[:,-1] = df.Year_Scaled * m                
    x2[:,:-1] = x
    
    # build KMeans model
    model = KMeans(n_clusters=n_clusters, random_state=42)
    predicted = model.fit_predict(x2)
    
    # get Silhouette score across all clusters
    silhouette_all = silhouette_score(x, predicted)
    result['Silhouette Score'] = silhouette_all
    
    # compute avg Silhouette score
    silhouette_avg = silhouette_samples(x, predicted)
    
    # compute metrics for each cluster
    avg_std = []
    std_tfidf = []
    avg_std_year = []
    avg_silhouette_by_std_year = []
    for i in range(n_clusters):
        j = i+1
        result[f'silhouette_clsuter_{j}'] = np.mean(silhouette_avg[predicted==i])
        avg_std.append(result[f'silhouette_clsuter_{j}'])
        std_tfidf.append(np.std(x[predicted==i]))
        avg_std_year.append(np.std(df.Year_Scaled[predicted==i]))
        avg_silhouette_by_std_year.append(result[f'silhouette_clsuter_{j}'] / avg_std_year[-1])
        result[f'std_year_cluster_{j}'] = avg_std_year[-1]
        result[f'silhouette_by_std_year_cluster_{j}'] = result[f'silhouette_clsuter_{j}'] / avg_std_year[-1]
        
    result['std_tfidf'] = np.mean(std_tfidf)
    result['avg_std'] = np.std(avg_std)
    result['avg_std_year'] = np.mean(avg_std_year)
    result['avg_silhouette_by_std_year'] = np.mean(avg_silhouette_by_std_year)
    
    return result

def get_clusters_timeline(predictions):
    '''
    Get timeline for each cluster
    
    Parameter:
    - predictions dataframe
    '''
    # get timeline for each cluster
    clusters_timeline = list(predictions.groupby(['Topic Id'])['Year'].unique())
    
    # a list of timeline string for each cluster
    clusters_timeline_str = []

    # iterate through the timeline for each cluster
    for years_list in clusters_timeline:
        min_year = []     # lower bound of timeline
        max_year = []     # upper bound of timeline
        
        # sort years_list
        years_list_sorted = np.sort(years_list)
        
        # build lower bound and upper bound for timeline
        for i, year in enumerate(years_list_sorted):
            if i == 0: 
                # set value for lower bound if this is the first item in years_list
                min_year.append(year)
            else:
                if len(max_year) == 0:
                    if year == min_year[len(min_year) - 1] + 1:
                        # if current year equals previous min year + 1
                        # set value for upper bound
                        max_year.append(year)
                    else:
                        # if current year is not an increment of previous min year (non-consecutive year)
                        max_year.append(0)      # set upper bound to zero to indicate a gap in the timeline
                        min_year.append(year)   # set value for the next sequence in the timeline (lower bound)
                elif len(min_year) > len(max_year) and year == min_year[len(min_year) - 1] + 1:
                    # if there exists a lower bound but no upper bound value
                    # and current year is an increment of the previous min_year
                    # set upper bound value to current year
                    max_year.append(year)
                elif len(min_year) > len(max_year) and year > min_year[len(min_year) - 1] + 1:
                    # if there exists a lower bound but no upper bound value
                    # and current year is NOT an increment of the previous min_year
                    # append 0 to max_year to indicate there is a gap in the timeline
                    max_year.append(0)
                    min_year.append(year)    # append current year to min_year
                elif len(min_year) == len(max_year) and year == max_year[len(max_year) - 1] + 1:
                    # if there exists a timeline for current year
                    # update the upper bound to current year
                    max_year[len(max_year) - 1] = year
                elif len(min_year) == len(max_year) and year > max_year[len(max_year) - 1] + 1:
                    # if current year is not an increment of the upper bound of the current timeline
                    # add year to the new lower bound timeline
                    min_year.append(year)

        # if len of lower bound and len of upper bound are not equal
        # set the last item in upper to zero to signify the end of timeline
        if len(min_year) > len(max_year):
            max_year.append(0)

        # iterate through min_year
        # and build timeline text string
        text = ''
        for j, yr in enumerate(min_year):
            if len(text) == 0:
                text = str(yr)
            else:
                text = text + ', ' + str(yr)

            if max_year[j] > 0:
                text = text + '-' + str(max_year[j])
        clusters_timeline_str.append(text)
    
    return clusters_timeline_str

def cal_cluster_bias(df, x_vector, terms_matrix_df, terms_label, bias=0.1, n_clusters=15):
    '''
    Perform biased clusterings on the data
    
    Paramers:
    - df: journal dataset
    - x_vector: vector representation of the data
    - terms_sparse_matrix: scipy sparse matrix of terms
    - terms_label: the label for each term in the sparse matrix
    - bias: the bias amount
    - n_clusters: the number of clusters
    '''
    
    # apply bias to x_vector
    x_vector_bias = np.zeros((x_vector.shape[0], x_vector.shape[1]+1))
    x_vector_bias[:,-1] = df.Year_Scaled*bias
    x_vector_bias[:,:-1] = x_vector
    
    # create biased clustering model
    model = SpectralClustering(n_clusters=n_clusters,random_state=42)
    predicted = model.fit_predict(x_vector_bias)
    predicted += 1

    # get metrics score
    score = silhouette_score(x_vector, predicted)
    silhouette_avg = silhouette_samples(x_vector, predicted)
    avg_silhoutte_per_cluster = []
    bias_avg_std_year = []
    trend_score = []
    for i in range(n_clusters):
        i+=1
        tmp_silhotte = np.mean(silhouette_avg[predicted==i])
        bias_avg_std_year.append(np.std(df.Year_Scaled[predicted==i]))
        avg_silhoutte_per_cluster.append({i:tmp_silhotte})
        trend_score.append(np.mean(tmp_silhotte / bias_avg_std_year[-1]))
       

    # get the top 20 extracted terms
    terms = get_top_keywords(predicted, 20, terms_matrix_df, terms_label)
    
    # build summary data frame
    df['Topic Id'] = predicted
    df_summary = pd.DataFrame()
    df_summary['Terms'] = terms
    df_summary['Trend Score'] = trend_score
    df_summary['Bias Avg Std Year'] =  bias_avg_std_year
    df_summary['Silhouette Score'] = [item[i+1] for i,item in enumerate(avg_silhoutte_per_cluster)]
    df_summary['Number of Articles'] = df.groupby(['Topic Id']).Abstract.count().values
    
    # compute percentage of Articles
    sum_articles = df_summary['Number of Articles'].sum()
    df_summary['Article %'] = df_summary['Number of Articles'].map(lambda x: (x/sum_articles)*100)
    df_summary['Topic Id'] = df_summary.index+1
    
    # get cluster's timeline
    df_summary['Timeline'] = get_clusters_timeline(df)
    
    # reorganize columns order for df_summary
    df_summary = df_summary[['Topic Id', 'Terms', 'Timeline', 'Number of Articles', 'Article %', 
                             'Trend Score', 'Silhouette Score', 'Bias Avg Std Year']]
    
    return df_summary, df