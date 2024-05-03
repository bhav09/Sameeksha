# Importing Dependencies

import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import traceback
import requests
import nltk
from PIL import Image
import time
from itertools import permutations

# CONSTANTS - credentials
f = open('credentials.json', 'r')
creds = json.load(f)
gemini_token = creds['gemini_api']

def generate_response_gemini_image(prompt, img):
        model_cv = genai.GenerativeModel('gemini-pro-vision')
        response = model_cv.generate_content([prompt, img], stream=True)
        response.resolve()
        return re.sub(r"\*\*([^*]+)\*\*", r"\1", response.text)

def generate_response(prompt, temperature, safety_setting):
    """
    Generates a resopnse by hitting to Gemini

    Parameters:
    - prompt (str): Description of the table.
    - temperature: The DataFrame containing the file data.

    Returns:
    - dict: Data dictionary containing the description of the table, each column, and its data type.
    """
    generation_config = {
      "temperature": temperature,
      "top_p": 1,
      "top_k": 1,
    }
    safety_settings = [
        {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": safety_setting
        },
        {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": safety_setting
        },
        {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": safety_setting
        },
        {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": safety_setting
        },
    ]
    genai.configure(api_key=gemini_token)
    model = genai.GenerativeModel('gemini-pro')
    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)
    convo = model.start_chat(history=[])
    convo.send_message(prompt)
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", convo.last.text)


def create_data_dictionary(table_description, df):
    """
    Create a data dictionary based on the table description, columns, and data types provided.

    Parameters:
    - table_description (str): Description of the table.
    - df: The DataFrame containing the file data.

    Returns:
    - dict: Data dictionary containing the description of the table, each column, and its data type.
    """

    # Prompts
    create_data_dict = f'''Table description: {table_description}
    Columns: {df.columns}
    Data types: {df.dtypes}

    Instruction:
    1. Based on the above mentioned details create a data dictionary which a small description of table, each column and the data type of each column.
    2. Don't generate anything else. Be concrete and concise in your response
    3. Give the output in the expected format of a dictionary only!
    '''
    output = '''
    Expected Output -> 
    data_dict={
    'tbl_description': 'description of table', 
    'columns': {
                'Name of the column 1': {'col_description':'description of column 1', 'data_type':'Data Type of the column 1'},
                'Name of the column 2': {'col_description':'description of column 2', 'data_type':'Data Type of the column 2'},
                'Name of the column 3': {'col_description':'description of column 3', 'data_type':'Data Type of the column 3'}
            }
    }
    '''

    create_data_dict += output
    response = generate_response(create_data_dict, 0, 'BLOCK_NONE')
    response = response.replace('`', '')
    # open('data_dictionary.txt','w+').write(response)
    d, data_dict = {}, {}
    d['data_dict'] = data_dict
    exec(response, d)
    # data_dict = open('data_dictionary.txt','r').read()
    return d['data_dict']

def dynamic_safety_setting(df):
    """
    Based on the data, dynamically adjusts its safety setting.

    Parameters:
    - df: The DataFrame containing the file data.

    Returns:
    - str: String containing safety setting
    """
    safety_setting = 'BLOCK_ONLY_HIGH'
    temperature = 0
    identify_threat_level = f'''
    Role: You are Gemini and a very helpful assisstant.

    Action: Based on harm categories identify the level of threat as: LOW or HIGH
    Table: {table_description}
    Data: {df.head()}
    Harm categories: HARM_CATEGORY_HARASSMENT, HARM_CATEGORY_SEXUALLY_EXPLICIT, HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_DANGEROUS_CONTENT.

    Instructions:
    1. Restrict your response to only LOW or HIGH at all costs
    2. If any of the Harm category is found then return HIGH at all costs!
    3. Don't include any other text. 

    Expected output format: LOW or HIGH etc.'''

    threat_level = generate_response(identify_threat_level, temperature, 'BLOCK_NONE')
    print(threat_level)
    if threat_level=='HIGH':
        print('Taking user consent..')
        safety_setting = 'BLOCK_NONE'
    print('Safety setting has been set to: ', safety_setting)
    return safety_setting

def drop_high_missing_columns(df, missing_threshold=25):
  """Drops columns in a pandas DataFrame that have more than the specified missing value threshold.

  Args:
      df (pd.DataFrame): The DataFrame to process.
      missing_threshold (float, optional): The threshold for the proportion of missing values in a column. Defaults to 0.1 (10%).

  Returns:
      pd.DataFrame: The DataFrame with columns exceeding the missing value threshold dropped.
  """

  # Calculate the percentage of missing values per column
  missing_vals = 100 * df.isnull().sum() / len(df)

  # Identify columns to drop
  cols_to_drop = missing_vals[missing_vals > missing_threshold].index

  # Drop the columns if any
  if len(cols_to_drop) > 0:
    return df.drop(cols_to_drop, axis=1)
  else:
    return df.copy()

def normalize_date_format(date_column):
    # Define possible date formats
    # Define the elements (year, month, day)
    elements = ['%Y', '%m', '%d']

    # Generate all permutations
    perms = permutations(elements)

    # Format permutations into strings
    possible_formats = ['/'.join(perm) for perm in perms]

    perms = permutations(elements)
    additional_formats = ['-'.join(perm) for perm in perms]
    possible_formats += additional_formats

    # Initialize an empty list to store normalized dates
    normalized_dates = []

    # Iterate over each date in the date column
    for date_str in date_column:
        # Initialize a variable to store the normalized date
        normalized_date = None
        # Iterate over each possible date format
        for date_format in possible_formats:
            try:
                # Try to parse the date using the current format
                normalized_date = pd.to_datetime(date_str, format=date_format).strftime("%Y-%m-%d")
                # If parsing succeeds, break the loop
                break
            except ValueError:
                # If parsing fails, continue to the next format
                continue
        # If no valid format was found, append None to the list
        if normalized_date is None:
            normalized_dates.append(None)
        else:
            # Otherwise, append the normalized date to the list
            normalized_dates.append(normalized_date)
    return normalized_dates

def auto_debugger(prompt, temperature, safety_setting):
    error_message = f'''
        You are an expert coder! Debug the below code for me.
        Code: {prompt}
        Traceback of the code: {traceback.format_exc()}

        Adhere to below instructions at all costs!
        Instruction:
        1. Identify the cause of the error and rewrite the code - make it error free
        2. Don't include any text in your response
        3. Rewrite the code as a function
        4. Follow these instructions by all means
        '''
    debugged_code = generate_response(error_message, temperature, safety_setting)
    return debugged_code

class RecommendAnalysis:
    def __init__(self, table_description, data_dict, df):
        """
        Initialize the RecommendAnalysis class with table description, data dict and a df.
        
        Parameters:
        - table_description (str): The description of the table
        - data_dict (dict): A data dictionary
        - df (dataframe): A pandas dataframe consisting data
        """
        self.table_description = table_description
        self.data_dict = data_dict
        self.df = df

    def recommend_analysis(self):
        """
        Read the file based on its type (CSV or Excel).
        
        Returns:
        - list: Returns a list of different analysis that can be performed on the data
        """
        prompt = f'''{self.table_description}\n Refer {self.data_dict['columns'].keys()} and tell me the different analysis that can be done from the available columns keeping the given use case in mind.
            Focus on Exploratory Data Analysis only. 

            Unique values in categorical column: {self.df.select_dtypes(include='object').nunique()}
            Unique values in numerical column: {self.df.select_dtypes(include='number').nunique()}

            Instructions:
            1. Keep your response consise and concrete
            2. Give your suggestions in bullet points
            3. Mention the columns that will help in the completion of the respective analysis
            4. Make the analysis rich by including as many important columns as possible. Don't include columns with no/very little variance.
            5. Every new type of analysis in the response should be represented with ">>" at all costs
            6. The analysis should be complex but at the same time either convey action items or actionable insights
            7. Don't generate a column name by your own. Use only the columns: {self.data_dict['columns'].keys()}
            8. Don't mention the columns in "" or ''.
            9. Don't add anything else to your response, except for Analysis name and the relevant columns
            10. Give the response in the expected format only.
            11. The analysis should be suggested from the growth/impact to the business - perspective


            Expected output:
            >> Analysis 1
            - Columns: A, B, C etc.

            >> Analysis 2
            - Columns: A, C, E etc.
            etc.
            '''
        
        types_of_analysis = generate_response(prompt, 0.5)
        list_of_analyses = types_of_analysis.split('>> ')[1:]
        return list_of_analyses

class Analysis:
    def __init__(self, my_analysis, df, data_dict, safety_setting):
        self.my_analysis = my_analysis
        self.my_analysis += "\nAlways Include: Relevant columns, numbers/figures associated with the analysis."
        self.df = df
        self.data_dict = data_dict
        self.safety_setting = safety_setting

    def identify_columns(self):
        identify_colums = f'''Analysis: {self.my_analysis}
            Remember: Almmost every analysis requires some kind of aggregation or grouping.
            First 5 rows of Dataframe for your reference: {self.df.head()}

            Instructions:
            1. Based on the Analysis mentioned, Give the names of the most relevant columns from {self.data_dict} by studying details about each column description.
            2. Don't generate any column(s) of your own
            3. If the analysis request is not direct then identify a logic from the given columns that would help you with the analysis.
            4. Don't write anything else, just the column names.

            Expected Output if relevant columns found:
            Columns: Col 1, Col 2, Col 3 etc.
            '''

        column_names = generate_response(identify_colums, 0, self.safety_setting)
        columns_index = column_names.find("Columns:")

        # Extract the text after "Columns:"
        columns_text = column_names[columns_index + len("Columns:"):].strip()

        # Print the extracted text
        columns = columns_text.split(', ')
        return columns

    def pre_dataprep(self, columns):
        columns = self.identify_columns()
        self.data = self.df[columns]
        self.columns_intel = ''
        # print(type(self.data_dict))
        for key, val in self.data_dict['columns'].items():
            if key in columns:
                self.columns_intel+=f'{key}: {val}\n'

        # Dropping columns with missing values
        self.data = drop_high_missing_columns(self.data, missing_threshold=25)

        # Checking the consistency of date columns
        fetch_column = f'''Refer columns info: {self.columns_intel}
            And tell which column refers to date.
            Instructions: Don't generate anything else but the column name.

            Expected output: if present then: Column name - else: not_found '''

        date_col = generate_response(fetch_column, 0, self.safety_setting)
        if date_col!= 'not_found':
            # Assuming df is your DataFrame and 'date_column' is the name of your date column
            self.data[f'{date_col}'] = normalize_date_format(self.data[f'{date_col}'])
            self.data[f'{date_col}'] = pd.to_datetime(self.data[f'{date_col}'])
        else:
            pass

        # To check which preprocessing template to use: numeric or text
        template_check = f'''
            Top 5 rows: {self.data.head()}
            Data: {self.data.columns}

            Based on the above details tell me what type of data is it?
            Rules:
            1. If consists text data then write 'Text'
            2. Else 'Numeric'
            3. Don't write anything else just respond whether it is 'Text' or 'Numeric'
            '''

        template_to_choose = generate_response(template_check, 0, self.safety_setting)
        return self.data, template_to_choose

    def prep_template(self, template_to_choose):
        # Prompt Template for numeric data - 
        if template_to_choose=='Numeric':
            self.preprocessing_template = '''
            > Data Imputation:-

                When to use: Data imputation is used to fill in missing values (e.g. Null, None or NaN) in the dataset. Impute mode for categorical and mean/median otherwise.
                For what type of data: This step is applicable to numerical and categorical data.

            > Feature Scaling:-

                When to use: Feature scaling is can be done when the features in the dataset have different scales. 
                Do not scale columns/features that are ordinal in nature like rank, ratings etc at any cost!
                For what type of data: This step is primarily applicable to numerical data, but it can also be used for some types of categorical data.

            > Noise Reduction:-

                When to use: Noise in the data can arise from various sources, such as measurement errors or data collection processes. Noise reduction techniques aim to remove or minimize the impact of noise on the dataset.
                For what type of data: This step is applicable to numerical data, and categorical data.

                Actions:
                For numerical data, apply smoothing techniques such as moving averages or median filters.
                For categorical data, grouping rare categories or merging similar categories can reduce noise.

            > Feature Engineering:-

                When to use: Feature engineering involves creating new features from existing ones or transforming existing features.
                Remember: Do it only when it would help in the analysis.
                For E.g 1 If column like date is involved then make sure the column has a consistent format i.e. "datetime format" - "YYYY-MM-DD" by all means!
                For what type of data: This step is applicable to all types of data.

                Actions:
                Generate new features by combining existing ones, extracting useful information from variables, or creating interaction terms.
                Transform features using mathematical functions such as logarithms, square roots, or polynomial transformations to better capture non-linear relationships.

            > Data Normalization or Standardization:-

                When to use: Normalization or standardization can be applied to scale numerical data to a standard range or distribution if required. 
                You shouldn't do it to columns that are ordinal in nature like rank, rating etc, educational level etc.
                For what type of data: This step is applicable to numerical data

                Actions:
                Scale numerical features to a specific range (e.g., [0, 1]) using min-max scaling or standardize features to have a mean of 0 and standard deviation of 1 using z-score normalization.

            > '''

            pattern = r'> (.*?):-'
            preprocessing_steps = re.findall(pattern, self.preprocessing_template)
            self.prep_details = self.preprocessing_template.split('>')[1:-1]

        if template_to_choose=='Text':
            # Prompt Template for Text data - 
            self.preprocessing_template = '''
            > Data Cleaning:-

                When to use: Should be performed to remove stop words or punctuation marks from text data.
                For what type of data: This step is applicable to textual data.
                
                Actions:
                Remove irrelevant information from text data, such as stop words or punctuation marks.

            > Data Imputation:-

                When to use: Data imputation is used to fill in missing values (e.g. Null, None or NaN) in the dataset. Impute mode for categorical and mean/median otherwise.
                For what type of data: This step is applicable to numerical and categorical data. Text data cleaning techniques can sometimes address missing values, but imputation might be necessary in specific cases.

            > Text Preprocessing:-

                When to use: Text preprocessing involves cleaning and transforming textual data into a format suitable for analysis. 
                Remember you need to do it only when the text is a sentence(s) and not for categorical data. Identify if the data is categorical or not.
                For what type of data: This step is specific to textual data, such as natural language text.

                Actions:
                Lowercase all text
                Apply stemming or lemmatization to reduce words to their root form (if applicable)

            > Noise Reduction:-

                When to use: Noise in the data can arise from various sources, such as measurement errors or data collection processes. Noise reduction techniques aim to remove or minimize the impact of noise on the dataset.
                For what type of data: This step is applicable to numerical data, textual data, and categorical data.

                Actions:
                For numerical data, apply smoothing techniques such as moving averages or median filters.
                For categorical data, grouping rare categories or merging similar categories can reduce noise.

            > Feature Engineering:-

                When to use: Feature engineering involves creating new features from existing ones or transforming existing features to improve the performance of machine learning models. For E.g. If a feature like date is involved and if the data is on a daily basis - aggregate it to weekly or monthly basis for better analysis unless not a stock price data.
                For what type of data: This step is applicable to all types of data.
                Remember: Do it only when it would help in the analysis.
                For E.g 1 If column like date is involved then make sure the column has a consistent format i.e. "datetime format" - "YYYY-MM-DD" by all means!
                For what type of data: This step is applicable to all types of data.

                Actions:
                Generate new features by combining existing ones, extracting useful information from text or categorical variables, or creating interaction terms.
                Transform features using mathematical functions such as logarithms, square roots, or polynomial transformations to better capture non-linear relationships.
                Apply techniques specific to text data, such as TF-IDF (Term Frequency-Inverse Document Frequency) to weight the importance of words.

            > Data Normalization or Standardization:-

                When to use: Normalization or standardization can be applied to scale numerical data to a standard range or distribution if required by the specific model being used. 
                You shouldn't do it to columns that are ordinal in nature like rank, rating etc, educational level etc.
                For what type of data: This step is applicable to numerical data and is optional depending on the model's requirements.

                Actions:
                Scale numerical features to a specific range (e.g., [0, 1]) using min-max scaling or standardize features to have a mean of 0 and standard deviation of 1 using z-score normalization.

            > '''

            pattern = r'> (.*?):-'
            preprocessing_steps = re.findall(pattern, self.preprocessing_template)
            self.prep_details = self.preprocessing_template.split('>')[1:-1]
        return preprocessing_steps
    
    def identifying_prep_steps(self, data, preprocessing_steps):
        # Checking preprocessing steps
        temperature = 0
        preprocessing_dict = {}
        self.data = data
        for idx, step in enumerate(tqdm(preprocessing_steps)):
            step_to_take = f'''
            Details -
            Analysis to perform: "{self.my_analysis}"
            Based on the analysis identify if preprocessing "{step}" is required or not
            Columns: {self.columns_intel}
            Data dypes: {self.data.dtypes}
            Description of data: {self.data.describe()}
            Preprocessing Details: {self.prep_details[idx]}
            Remember: Almost all the type of analysis include aggregation/grouping of data. Based on that identify whether {step} preprocessing step is necessary or not.

            Adhere to below instructions at all costs!
            Instructions -
            0. Consider the details shared above to make the rules for your preprocessing test if needed
            1. Assume the dataframe "self.data" exists already
            2. Do not read data from anywhere
            3. Write a simple error free code
            4. Write a function that performs the preprocessing test and returns the response of the function in 'True' or 'False'
            5. Write only the code, don't include any other text/explanation in header or footer at any cost.
            6. Install and Import whatever package is necessary
            7. Keep the original dataframe intact. Don't overwrite it - at any cost
            8. If preprocessing step is not applicable for the data mentioned then return 'False'

            Expected Output:
            def preprocessing_test(data):
                # Preprocessing logic

                return True or False based on the logic
            result = preprocessing_test(data)
            '''
            count = 0
            data = self.data
            # Automated debugging
            while count<2:
                try:
                    if count==0:
                        test_of_step = generate_response(step_to_take, temperature, self.safety_setting)
                    test_of_step = test_of_step.replace('python', '')
                    test_of_step = test_of_step.replace('`', '')
                    d = {}
                    d['test_of_step'] = test_of_step
                    d['data'] = data
                    # print(test_of_step)
                    exec(test_of_step, d)
                    preprocessing_dict[step] = d['test_of_step']
                    # self.data = data
                    break
                    
                except Exception as e:
                    temperature += 0.2
                    test_of_step = auto_debugger(test_of_step, temperature, self.safety_setting)
                    count+=1
        # self.data = data
        return self.data, preprocessing_dict
    
    def perform_preprocessing(self, preprocessing_dict):
        # Performing only those preprocessing steps that are required

        # to know how was preprocessing done - code_transcript
        self.code_transcript = ''
        temperature = 0
        for key, val in tqdm(preprocessing_dict.items()):
            if val==True:
                write_code_for_prep_step = f'''
                Details -
                    Analysis to perform: {self.my_analysis}
                    Preprocessing step: {key}
                    Preprocessing Details: {re.findall(rf'> {key}:-(.*?)>', self.preprocessing_template, re.DOTALL)[0]}
                    Columns: {self.columns_intel}
                    Description of data: {self.data.describe()}
                    Data types of columns: {self.data.dtypes}
                
                Adhere to below instructions at all costs!
                Instructions -
                0. Consider the details shared above for rules of your preprocessing test if required
                1. Assume the dataframe "self.data" exists already
                2. Do not read or generate data by yourself
                3. Do not mention python language in your response
                4. Write simple code that's easy to understand without any errors
                5. Write a function that performs the preprocessing and return the dataframe after preprocessing it
                6. Only write the code don't include any other text. The code shouldn't have any error be syntactical or logical
                7. Call the function. Make sure you don't return an empty dataframe.
                8. Don't use lambda function to write your code at any cost!
                9. From the function name it should be understandable which preprocessing technique was used.

                Expected output:
                def some_function_name():
                    # Some logic

                    return some_value
                    
                # Calling function
                self.data = some_function_name()
                '''
                count = 0
                data = self.data
                # Automated debugging
                while count<2:
                    try:
                        if count==0:
                            prep_code_output = generate_response(write_code_for_prep_step, temperature, self.safety_setting)
                        prep_code_output = prep_code_output.replace('`','')
                        prep_code_output = prep_code_output.replace('python','')
                        d={}
                        d['prep_code_output'] = prep_code_output
                        d['data'] = data
                        exec(prep_code_output, d)
                        self.data = data
                        break
                    
                    except Exception as e:
                        temperature += 0.2
                        prep_code_output = auto_debugger(prep_code_output, temperature, self.safety_setting)
                        count+=1
                self.code_transcript+=prep_code_output+'\n-----------------------------------------\n'
        return self.data
    
    def perform_analysis(self):

        # Perform analysis - 
        # print(self.my_analysis)
        write_code_for_analysis = ''
        count, temperature = 0, 0
        while count<2:
            try:
                query = f'''
                Analysis to perform: {self.my_analysis}
                Remember: Analysis is always some type of aggregation or grouping of certain columns to get the desired result. So perform aggregation/grouping
                at all costs!.

                Instructions:
                1. Write code in python to execute the analysis - at all costs.
                2. Assume a dataframe with the name "self.data" already exists.
                3. Dataframe df has the following columns: {self.data.columns}. Use the column names for your refernece while generating the code.
                4. Don't include the code to read the file. Write the code assuming the dataframe already exists.
                5. Don't generate your own data. 
                6. First 5 rows of the dataframe you will work on: {self.data.head()}
                7. Dataframe should have {self.data.columns} as its columns only.
                8. Don't write code to train any machine learning model. Write code only to perform the analysis
                9. Aggregate/Group the dataframe "self.data" to get the desired result for Analysis by all means!
                10. Write code only the way shown below. And call the function analysis() by all means!

                Expected output - I need output in the similar fashion only!
                def analysis(data):
                    # Some Logic

                    return some_value

                # Calling the function
                data = analysis(data)
                '''
                data = self.data
                if count==0:
                    write_code_for_analysis = generate_response(query, temperature, self.safety_setting)
                write_code_for_analysis = write_code_for_analysis.replace('python', '')
                write_code_for_analysis = write_code_for_analysis.replace('`','')
                d = {}
                if 'data = analysis(data)' not in write_code_for_analysis:
                    function_call = '\ndata = analysis(data)'
                    write_code_for_analysis += write_code_for_analysis+function_call
                # print(write_code_for_analysis)
                d['write_code_for_analysis'] = write_code_for_analysis
                d['data'] = data
                exec(write_code_for_analysis, d)
                # self.data = analysis(self.data)
                self.data = d['data']
                break
            except Exception as e:
                temperature += 0.2
                write_code_for_analysis = auto_debugger(write_code_for_analysis, temperature, self.safety_setting)
                count+=1

            self.code_transcript+=write_code_for_analysis+'\n-----------------------------------------\n'
        # print(self.code_transcript)
        print(d['data'])
        return self.data, self.code_transcript

class GenerateInsights:
    def __init__(self, my_analysis, data, analysis_file, safety_setting, code_transcript) :
        self.my_analysis = my_analysis
        self.data = data
        self.analysis_file = analysis_file
        self.safety_setting = safety_setting
        self.code_transcript = code_transcript
    
    def insight_type_identification(self):
        # Insight type identification
        self.analysis_output = open(self.analysis_file+'.csv').read()
        insight_prompt = f'''
        Based on the Analysis Output shared below, tell what would be best way to represent the insights of the given analysis - Visualization or Text
        1. Choose Visualization when the number of fields/columns are less but more than one - and thus the chart formed would be readable to user.
        2. Choose Text when the number of values are more or the output length is long.

        Expected Output: Visualization or Text
        Analysis wanted: {self.my_analysis}
        Analysis Output: {self.analysis_output}
        '''
        print(self.safety_setting)
        insight_choice = generate_response(insight_prompt, 0.2, self.safety_setting)
        return insight_choice

    def understand_image(self, img):
        prompt = f'''
            Analysis requested: {self.my_analysis}
            Analysis Output: {self.analysis_output}
            Data: {self.data}

            The given image is extracted from the analysis. It is a type of visualisation. 
            If visualization: 
                1. Identify the type of visualization
                2. Using labels and legends extract important and accurate insights with numerical figures or percentages from the visualization if there are any.
                3. The insights should be interesting, accurate and actionable - related to the analysis mentioned.
            
            Instructions:
            1. Make sure above conditions are met.
            2. Do not include any irrelevant insight/text in your response. 
            3. Be concise, crisp and concrete. Write insights creatively. Each new insight shouldn't start the same way. Make every insight's beginning look unique.
            4. Refer output analysis to generate actionable insights based on the analysis asked and give business related suggestion if asked.
        '''
        return generate_response_gemini_image(prompt, img)
    
    def generate_insights(self, table_description):
        insight_choice = self.insight_type_identification()
        if insight_choice=='Visualization':
            count, temperature = 0, 0
            while count<2:
                try:
                    visualization_prompt = f'''
                        Information - 
                        Table: {table_description}
                        Task: {self.my_analysis}
                        Output: {self.analysis_output}

                        TYPES OF CHARTS:
                        1. Line Chart: Good for trends over time/categories, bad for many data points or complex relationships.
                        2. Bar Chart: Compares categories/frequencies, avoid for many categories or negative values.
                        3. Scatter Plot: Explores relationships between two variables, not ideal for more than 3 variables or unclear patterns.
                        4. Pie Chart: Shows proportions/contribution of a whole, avoid for many categories or unclear comparisons.
                        5. Histogram: Visualizes distribution of continuous data, not for categorical data.
                        6. Box Plot: Compares distributions across categories, avoid if outliers dominate.
                        7. Heatmap: Good for visualizing relationships between many variables, bad for complex data, overwhelming for large datasets
                        8. Word cloud: good for visual exploration of frequent terms in text data, bad for in-depth analysis.
                        9. Network Graph: Shows connections between entities (e.g., social networks, protein interactions), Not ideal for large or dense networks.
                        10. Sankey Diagram: Tracks flows across stages in a process (e.g., customer journeys, material flow). Gets messy with many stages or branches.
                        11. Choropleth Mapbox: Colors geographic regions like country etc based on a data value (e.g., election results, population density). Avoids if data varies greatly within regions.
                        12. Heatmap (Geographic): Colors geographic areas based on data intensity. Overwhelming for cluttered data or small regions.
                        13. Flow Map: Shows movement between geographic locations (e.g., migration patterns, trade routes). Can get confusing with many flows or overlapping paths.

                        DEFAULT CHARTS -:
                        1. Trends over time/categories: Line chart
                        2. Compare categories/frequencies: Bar Chart
                        3. Compare frequency but also has regions like countries involved: Choropleth Mapbox plot - using plotly.graph_objects;
                        4. To show proportions: Pie chart
                        5. Comparing distribution: Box plot
                        6. Visualizing distribution of continous data: Histogram (Geographic) or Choropleth
                        7. Exploration of frequent/common terms in text data: Word cloud

                        Follow the instructions by all means.
                        Instructions -
                        0. Based on the info available above identify what type of chart would suit the best to convey the insights for "{self.my_analysis}" - consider readability of the chart as well.
                        1. Write code in python to perform an insightful visualization from the output shared to plot it - call the function at costs. Don't write any other text. Just code.
                        2. Make a new dataframe which has the following data: {self.analysis_output} and from '{self.analysis_file}.csv'
                        3. Don't generate your own data. Don't equate visualization with "self.data" variable at any cost.
                        4. Visualization should have title, axis labels, legend etc.
                        5. Save the visualization with the name '{self.analysis_file}.png' and '{self.analysis_file}.html' as well at all costs in the function itself.
                        6. Always show x axis labels with a rotation of 90 degrees if the number of labels are more than 8 else 60 degrees by all means.
                        7. If the chart can be built using Matplotlib, Seaborn, Geopandas then use it
                        8. Refer Code trascript: {self.code_transcript} to write an error free code.
                        9. If number of entities/rows representing are more then plot only the first/top 10 rows (and mention it in the graph that you have done it)
                        10. Don't make charts in black or white colour. Make the charts colourful.

                        Expected output:
                        def data_visualization(data):
                            # Some Logic
                            Write the code here

                            # Code to plot and show the visualization
                            Write the code here

                            # Code to save the visualization/figure/chart with name "{self.analysis_file}.png" and "{self.analysis_file}.html"
                            Saving the image in "{self.analysis_file}.png" and "{self.analysis_file}.html" based on INSTRUCTIONS shared above

                        # calling the function by all means
                        data_visualization(data)
                        '''
                    
                    vis_code = ''
                    data = self.data
                    if count==0:
                        vis_code = generate_response(visualization_prompt, temperature, self.safety_setting)
                    print(1)
                    vis_code = vis_code.replace('python', '')
                    vis_code = vis_code.replace('`', '')
                    # vis_code += '\ndata_visualization(data)'
                    d = {}
                    d['vis_code'] = vis_code
                    d['data'] = data
                    print(d['vis_code'])
                    exec(vis_code, d)
                    self.data = d['data']
                    vis_code = d['vis_code']
                    print('Executed')
                    break

                except Exception as e:
                    temperature += 0.2
                    vis_code = auto_debugger(vis_code, temperature, self.safety_setting)
                count+=1

            print(self.code_transcript)
            img = Image.open(f'{self.analysis_file}.png')
            insights = self.understand_image(img)
            print(insights)
            self.code_transcript += d['vis_code']+'\n-----------------------------------------\n'

        if insight_choice=='Text':
            textual_insight = f'''
                                Action: Read the analysis output of {self.my_analysis} carefully: {self.analysis_output}

                                Instructions:
                                1. Share the results and give concrete and crisp actionable or interesting insights based on the analysis output - if there are any.
                                2. Tone: Professional
                                3. Talk always in terms of numbers/figures or percentages
                                4. Don't generate data/insights of your own at any cost.
                                5. Provide only the insights that are required.'''

            insights = generate_response(textual_insight, 0.5, self.safety_setting)
            print(insights)
        return self.code_transcript, insights

class VyuEngine:
    def __init__(self, job):
        self.job = job
    
    def start_engine(self):
        index  = 0
        start = time.time()
        # Reading file
        df = self.job.df

        # Creating data dictionary
        table_description = self.job.table_description
        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        data_dict = create_data_dictionary(table_description, df)
        index+=1

        # Dynamic safety setting
        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        safety_setting = dynamic_safety_setting(df)
        index+=1
        print()

        # Recommend Analysis
        click = False
        if click==True:
            rec_anal_obj = RecommendAnalysis(table_description, data_dict, df)
            list_of_analyses = rec_anal_obj.recommend_analysis()
            print(list_of_analyses)
        
        # Initialising Analysis
        my_analysis = self.job.input_prompt
        anal_obj = Analysis(my_analysis, df, data_dict, safety_setting)

        # Pre data prep
        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        columns = anal_obj.identify_columns()
        index+=1
        print(columns)

        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        data, template_to_choose = anal_obj.pre_dataprep(columns)
        index+=1
        print()

        # Identifying Preprocessing Steps
        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        preprocessing_steps = anal_obj.prep_template(template_to_choose)
        data, preprocessing_dict = anal_obj.identifying_prep_steps(data, preprocessing_steps)
        index+=1

        # Performing Preprocessing and Analysis
        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        data = anal_obj.perform_preprocessing(preprocessing_dict)
        index+=1

        progress_bar.progress((index) / len(progress_text), text=progress_text[index])
        data, code_transcript = anal_obj.perform_analysis()
        index+=1

        analysis_file_name = str(pd.Timestamp.now()).replace(' ', '')
        data.to_csv(analysis_file_name+'.csv', index="False")
        progress_bar.progress((index+1) / len(progress_text), text=progress_text[index])
       
        gen_insights = GenerateInsights(my_analysis, data, analysis_file_name, safety_setting, code_transcript)
        code_transcript, insights = gen_insights.generate_insights(self.job.table_description)
        print('Execution Time: (in mins)',(time.time()-start)/60)
        
        self.job.output_file_name = analysis_file_name
        self.job.output_insights = insights
        return self.job

class Job:
    def __init__(self, input_prompt, df, table_description, output_insights, output_file_name):
        self.input_prompt = input_prompt
        self.df = df
        self.table_description = table_description
        self.output_insights = output_insights
        self.output_file_name = output_file_name

# MAIN
st.title('Sameeksha - The AI Data Analyst')
changes = '''
<style>
[data-testid = "stAppViewContainer"]
    {
    background-image:url('https://i.postimg.cc/7LLxyX4M/ai-hack.png');
    background-size:cover;
    }
    
    div.esravye2 > iframe {
        background-color: transparent;
    }
</style>
'''
st.markdown(changes, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=['csv'])
show_data = st.button('Show Data')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if show_data:
        if uploaded_file.name.endswith('csv'):
            st.write(df.head())

    table_description = st.text_input('Describe your Data')
    my_analysis = st.text_input('Describe your Analysis')
    start_analysis = st.button('Start Analysis')

    if start_analysis:
        progress_text = ['Creating data dictionary..', 'Dynamically setting up safety layers..',
                         'Understanding data..', 'Identifying preprocessing steps..', 'Preprocessing data..',
                         'Performing analysis..', 'Generating Insights..', 'Analysis Completed!']
        progress_bar = st.progress(0)
        output_insights, output_file_name = '', ''
        job = Job(my_analysis, df, table_description, output_insights, output_file_name)

        vyu = VyuEngine(job)
        
        job = vyu.start_engine()
        print(job.output_file_name)
        png_file = job.output_file_name+'.png'
        html_file = job.output_file_name+'.html'
        st.image(png_file)
        st.write(job.output_insights)

        
        if input('Press Y to delete the files:')=='Y':
            os.remove(f'{job.output_file_name}.csv')
            os.remove(f'{job.output_file_name}.html')
            os.remove(f'{job.output_file_name}.png')