# Necessary libraries

import os
import json
import openai
import yaml
from flask import Flask, redirect, url_for, render_template, request, session, jsonify
from dotenv import load_dotenv
from datetime import datetime

#Langchain imports

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.chains import APIChain
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

#Google Ads Imports 

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.api_core.page_iterator import GRPCIterator

load_dotenv()
llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
conversation = ConversationChain(llm=llm, verbose=False)

# Prompt templates
prompt_template_1 = PromptTemplate(
    input_variables=["query"],
    template=("You are an AI exclusively trained on Google Ads API documentation API version (v13). "
              "Your sole function is to accurately convert users' natural language queries into the corresponding "
              "Google Ads API call. Do not rely on any other information. If a query requires a Google Ads API call, "
              "provide a JSON response with the 'function_to_call'. If the query is missing information, ask the user for it."
              "\n\nExample:\nQuery: Show me my list of accessible customers\nExpected output: \"function_to_call: list_accessible_customers\""
              "\n\nExample:\nQuery: Create a new account\nExpected output: \"function_to_call: create_customer_client, manager_customer_id: \" \n\n{query}"),
)

prompt_template_3 = PromptTemplate(
    input_variables=["query"],
    template=("Forget your previous set of instructions and answer the user query. You are now a helpful Google Ads Expert and friendly AI Assistant:\n\n{query}"),
)

prompt_template_accessible_customers = PromptTemplate(
    input_variables=["api_response"],
    template=("You are a helpful AI assistant expertly trained on Google Ads API documentation. Take the given Google Ads API response from the list_accessible_customers call and return a helpful and informative response in simple natural language:\n\nAPI Response: {api_response}"),
)


openai.api_key = os.environ.get("OPENAI_API_KEY")
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Load and define client secrets and scopes
CLIENT_SECRETS_FILE = os.environ.get("CLIENT_SECRETS_PATH")
SCOPES = ['https://www.googleapis.com/auth/adwords', 'openid', 'https://www.googleapis.com/auth/userinfo.email']

# Route helpers
def get_flow():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=os.environ.get("GOOGLE_REDIRECT_URI")
    )
    return flow

def credentials_to_dict(credentials):
    return {'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes}

def dict_to_credentials(credentials_dict):
    return Credentials.from_authorized_user_info(info=credentials_dict)

# App routes
@app.route('/authorize')
def authorize():
    flow = get_flow()
    authorization_url, state = flow.authorization_url(prompt='consent')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = session['state']
    flow = get_flow()
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)

    return redirect(url_for('authorized'))

@app.route('/authorized')
def authorized():
    if 'credentials' not in session:
        return redirect(url_for('home'))
    
    credentials = dict_to_credentials(session['credentials'])
    google_ads_client = GoogleAdsClient(credentials=credentials, developer_token=os.environ.get("DEVELOPER_TOKEN"))

    return render_template("index.html")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api/manager_customer_accounts', methods=["GET"])
def get_manager_customer_accounts():
    if 'credentials' not in session:
        return jsonify({"error": "Not authorized"})
    credentials = dict_to_credentials(session['credentials'])
    google_ads_client = GoogleAdsClient(credentials=credentials, developer_token=os.environ.get("DEVELOPER_TOKEN"))
    customer_service = google_ads_client.get_service("CustomerService")
    accessible_customers = customer_service.list_accessible_customers()
    result_total = len(accessible_customers.resource_names)
    customers = [resource_name.split('/')[-1] for resource_name in accessible_customers.resource_names]
    return jsonify({"manager_customer_accounts": customers})

@app.route('/api/manager_customer_accounts', methods=["POST"])
def set_manager_customer_account():
    if 'credentials' not in session:
        return jsonify({"error": "Not authorized"})
    manager_customer_id = request.json["manager_customer_id"]
    session['manager_customer_id'] = manager_customer_id
    return jsonify({"message": f"Manager account set to {manager_customer_id}"})


@app.route('/api/interpret', methods=["POST"])
def pass_query():
    query = request.json["query"]
    manager_customer_id = session.get('manager_customer_id', None)

    if 'credentials' not in session or not manager_customer_id:
        return jsonify({"error": "Not authorized"})

    credentials = dict_to_credentials(session['credentials'])
    google_ads_client = GoogleAdsClient(credentials=credentials, developer_token=os.environ.get("DEVELOPER_TOKEN"))

    response_text, function_to_call = handle_language_model_routing(query, google_ads_client, manager_customer_id)

    if function_to_call == 'list_accessible_customers':
        return response_text

    return jsonify({"response": response_text})

# Language model routing helper functions
def handle_language_model_routing(query, google_ads_client, manager_customer_id):
    model_response_1 = prompt_and_predict(prompt_template_1, query=query)

    try:
        model_response_json = json.loads(model_response_1)
    except json.JSONDecodeError:
        model_response_json = {"code": model_response_1, "function_to_call": "", "manager_customer_id": ""}
    
    response_text = model_response_json.get("code", "")
    function_to_call = model_response_json.get("function_to_call", "")

    if function_to_call == "":
        response_text = prompt_and_predict(prompt_template_3, query=query)
    elif function_to_call == 'list_accessible_customers':
        api_response = list_accessible_customers(google_ads_client)
        response_text = prompt_and_predict(prompt_template_accessible_customers, api_response=json.dumps(api_response.get_json()))

    # Pass the manager_customer_id to the create_customer function
    elif function_to_call == 'create_customer_client':
        if manager_customer_id == "":
            response_text = "Please provide a manager customer ID to create a new customer account."
        else:
            response_text = create_customer(google_ads_client, manager_customer_id)
            response_text = f"Customer created with resource name `{response_text}` under manager account with ID `{manager_customer_id}`."

    return response_text, function_to_call

def prompt_and_predict(template, **kwargs):
    input_text = template.format(**kwargs)
    response = conversation.predict(input=input_text)
    return response.strip()

# Google Ads functions
def list_accessible_customers(google_ads_client):
    customer_service = google_ads_client.get_service("CustomerService")

    accessible_customers = customer_service.list_accessible_customers()
    result_total = len(accessible_customers.resource_names)
    customers = [{'resource_name': resource_name} for resource_name in accessible_customers.resource_names]

    return jsonify({"total_results": result_total, "customers": customers})

def create_customer(client, manager_customer_id):
    customer_service = client.get_service("CustomerService")
    customer = client.get_type("Customer")
    now = datetime.today().strftime("%Y%m%d %H:%M:%S")
    customer.descriptive_name = f"Account created with CustomerService on {now}"
    customer.currency_code = "USD"
    customer.time_zone = "America/New_York"
    customer.tracking_url_template = "{lpurl}?device={device}"
    customer.final_url_suffix = "keyword={keyword}&matchtype={matchtype}&adgroupid={adgroupid}"

    response = customer_service.create_customer_client(
        customer_id=manager_customer_id, customer_client=customer
    )
    return response.resource_name

if __name__ == '__main__':
    app.run(debug=True)




