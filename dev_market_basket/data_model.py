
from helper_functions import  *

#Read in one month of customer data
# Processing should go on SQL side for prod
path_string =  os.getcwd()+'/sql/'
sql_file= open(
    path_string + 'mvc_june_2020.sql', 'r').read()
df = pd.read_gbq(
        sql_file,
        project_id='infusionsoft-looker-poc',
        dialect='standard'
    )


# Add cvm flag columns
df_new, new_column_list=make_cvm_flags(df)


GROUP_BY_LIST=['app_name', 'ui']


#Summarize total contacts added over timeframe
df_contacts=df_new.groupby(GROUP_BY_LIST).agg(
    total_contacts_added=('total_contacts_added', 'sum')
).reset_index()


df_contacts['add_10_or_more']=np.where(df_contacts.total_contacts_added>=10, 1, 0)


#max values for new columns
df_max= df[new_column_list+GROUP_BY_LIST].groupby(GROUP_BY_LIST).max().reset_index()

## common column is app_name, should merge on that
df_summary = df_contacts.merge(df_max, how='left')


CVM_COLUMN_LIST=new_column_list+['add_10_or_more']



#ADD UI HERE
df_cvm1 = pd.DataFrame({'cvm_col1':CVM_COLUMN_LIST, "key":0})
df_cvm2 = pd.DataFrame({'cvm_col2':CVM_COLUMN_LIST, "key":0})
df_cvm= df_cvm1.merge(df_cvm2, how='outer')
#df_cvm3=pd.DataFrame({'ui':list(df.ui.unique()), "key":0})
#df_cvm=df_cvm.merge(df_cvm3)
df_cvm= df_cvm.drop('key', axis=1)





df_cvm['pmi']=df_cvm.apply(lambda x: get_pmi(df_summary,x.cvm_col1, x.cvm_col2, x.ui), axis=1)
df_cvm['npmi']=df_cvm.apply(lambda x: get_npmi(df_summary,x.cvm_col1, x.cvm_col2, x.ui), axis=1)
df_cvm['joint_prob']=df_cvm.apply(lambda x: joint_prob(df_summary,x.cvm_col1, x.cvm_col2, x.ui), axis=1)
df_cvm['marginal_prob_col1']=df_cvm.apply(lambda x: marginal_prob(df_summary,x.cvm_col1, x.ui), axis=1)
df_cvm['conditional_given_col1']=df_cvm.joint_prob/df_cvm.marginal_prob_col1









