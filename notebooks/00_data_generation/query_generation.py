# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Query Generator
# MAGIC
# MAGIC Notebook to generate synthetic queries to test telco support agent endpoint.
# MAGIC Can be scheduled as a Databricks job to continuously simulate customer interactions.

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q
# MAGIC %pip install mlflow databricks-agents --upgrade --pre
# MAGIC %pip install retry
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import mlflow
from retry import retry
from mlflow.deployments import get_deploy_client
from mlflow.entities.assessment import AssessmentSourceType, AssessmentSource, AssessmentError

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if root_path:
    sys.path.append(root_path)

from telco_support_agent.data.config import CONFIG

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("experiment_name", "/telco_support_agent/dev/experiments/dev_telco_support_agent")
dbutils.widgets.dropdown("include_multi_domain", "true", ["true", "false"])

# COMMAND ----------

env = dbutils.widgets.get("env")
AGENT_ENDPOINT_NAME = f"{env}-telco-customer-support-agent"
LLM_ENDPOINT = "databricks-claude-sonnet-4"
MAX_WORKERS = 5  # parallel query execution
QUERIES_PER_BATCH = 50  # number of queries to generate per execution
include_multi_domain_str = dbutils.widgets.get("include_multi_domain")
INCLUDE_MULTI_DOMAIN = include_multi_domain_str.lower() == "true"

# Customer ID generation based on data generation config
CUSTOMER_ID_START = 10001
CUSTOMER_COUNT = CONFIG["volumes"]["customers"]  # 1500 customers
CUSTOMER_ID_END = CUSTOMER_ID_START + CUSTOMER_COUNT - 1

# generated human agents for feedback capture
AGENT_NAMES = [
    "Sarah Chen", "Michael Rodriguez", "Jessica Williams", "David Park",
    "Emily Johnson", "Robert Taylor", "Amanda Davis", "James Wilson",
    "Maria Garcia", "Christopher Lee", "Ashley Brown", "Daniel Kim",
    "Lisa Anderson", "Kevin Martinez", "Rachel Thompson", "Brandon White",
]

print(f"Customer ID range: CUS-{CUSTOMER_ID_START:05d} to CUS-{CUSTOMER_ID_END:05d}")
print(f"Total possible customers: {CUSTOMER_COUNT}")

deploy_client = get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Response Formatting Utilities

# COMMAND ----------

class ResponseFormatter:
    """Utility class for formatting agent responses."""
    
    @staticmethod
    def extract_assistant_message(response: Dict[str, Any]) -> str:
        """Extract the main assistant message from the response."""
        if not response or 'output' not in response:
            return "No response content found"
        
        for output_item in response['output']:
            if output_item.get('type') == 'message' and output_item.get('role') == 'assistant':
                if 'content' in output_item:
                    for content in output_item['content']:
                        if content.get('type') == 'output_text':
                            return content.get('text', '')
        
        return "No assistant message found"
    
    @staticmethod
    def extract_function_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function calls made by the agent."""
        function_calls = []
        
        if not response or 'output' not in response:
            return function_calls
        
        for output_item in response['output']:
            if output_item.get('type') == 'function_call':
                function_calls.append({
                    'name': output_item.get('name', '').split('__')[-1],  # Get short name
                    'call_id': output_item.get('call_id'),
                    'arguments': output_item.get('arguments', '{}')
                })
        
        return function_calls
    
    @staticmethod
    def extract_custom_outputs(response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract custom outputs from the response."""
        return response.get('custom_outputs', {})
    
    @staticmethod
    def format_response_summary(response: Dict[str, Any], execution_time: float) -> str:
        """Create a formatted summary of the agent response."""
        if not response:
            return "❌ No response received"
        
        assistant_message = ResponseFormatter.extract_assistant_message(response)
        function_calls = ResponseFormatter.extract_function_calls(response)
        custom_outputs = ResponseFormatter.extract_custom_outputs(response)
        
        summary_lines = []
        summary_lines.append(f"Execution Time: {execution_time:.2f}s")
        
        routing_info = custom_outputs.get('routing', {})
        if routing_info:
            agent_type = routing_info.get('agent_type', 'unknown')
            summary_lines.append(f"Routed to: {agent_type.upper()} agent")
        
        customer_id = custom_outputs.get('customer')
        if customer_id:
            summary_lines.append(f"Customer: {customer_id}")
        
        if function_calls:
            summary_lines.append(f"Function Calls: {len(function_calls)}")
            for fc in function_calls:
                summary_lines.append(f"   • {fc['name']}")
        
        if assistant_message:
            summary_lines.append(f"Response: {assistant_message}")
        
        return "\n".join(summary_lines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query Generation Templates / Logic

# COMMAND ----------

@dataclass
class QueryContext:
    """Context for generating queries."""
    category: str
    requires_customer_id: bool
    requires_dates: bool
    persona_context: str
    business_scenario: str

QUERY_TEMPLATES = {
    "account": {
        "contexts": [
            QueryContext("account", True, False, "existing customer", "routine account inquiry"),
            QueryContext("account", True, False, "loyal customer", "subscription verification"),
            QueryContext("account", True, False, "family plan customer", "account details review"),
            QueryContext("account", True, False, "business customer", "service status check"),
            QueryContext("account", True, False, "new customer", "account setup verification"),
        ],
        "base_scenarios": [
            "customer wants to know their current plan details",
            "customer needs to verify their subscription status",
            "customer is asking about autopay settings",
            "customer wants to know when their account was created",
            "customer needs loyalty tier information",
            "customer is checking contract length and renewal dates",
            "customer wants to see all active subscriptions",
            "customer is asking about account segment and benefits",
            "customer needs to verify billing preferences",
            "customer wants to check if they qualify for upgrades"
        ]
    },
    
    "billing": {
        "contexts": [
            QueryContext("billing", True, True, "concerned customer", "bill inquiry"),
            QueryContext("billing", True, True, "budget-conscious customer", "payment planning"),
            QueryContext("billing", True, False, "confused customer", "charge explanation"),
            QueryContext("billing", True, True, "data-heavy user", "usage verification"),  # Enhanced
            QueryContext("billing", True, True, "traveling customer", "roaming charges"),
            QueryContext("billing", True, True, "usage-monitoring customer", "usage tracking"),
            QueryContext("billing", True, True, "business customer", "usage reporting"),
            QueryContext("billing", True, True, "family plan customer", "usage analysis"),
        ],
        "base_scenarios": [
            "customer sees unexpected charges on their bill",
            "customer wants to know when payment is due",
            "customer needs breakdown of current month charges",
            "customer is questioning data usage amounts", 
            "customer wants payment history for tax purposes",
            "customer needs to understand prorated charges",
            "customer is asking about auto-pay status",
            "customer wants to dispute a specific charge",
            "customer needs usage details for expense reporting",
            "customer is planning data usage for upcoming month",
            "customer wants to know their data usage for a specific month",
            "customer needs to check voice minutes used in last billing cycle", 
            "customer is asking about SMS usage over a date range",
            "customer wants to compare usage between different months",
            "customer needs total usage breakdown for expense reporting",
            "customer is checking if they're approaching data limits",
            "customer wants to analyze usage patterns over time",
            "customer needs usage details for a specific billing period",
            "customer is tracking usage to optimize their plan",
            "customer wants to know peak usage periods",
            "customer needs usage data for tax deduction purposes",
            "customer is monitoring family member usage on shared plan",
        ]
    },
    
    "tech_support": {
        "contexts": [
            QueryContext("tech_support", True, False, "frustrated customer", "service disruption"),
            QueryContext("tech_support", True, False, "tech-savvy customer", "configuration issue"),
            QueryContext("tech_support", True, False, "elderly customer", "device setup help"),
            QueryContext("tech_support", True, False, "business customer", "connectivity problem"),
            QueryContext("tech_support", True, False, "traveling customer", "roaming setup"),
        ],
        "base_scenarios": [
            "customer's phone won't connect to network",
            "customer has slow data speeds",
            "customer can't receive calls but can make them",
            "customer needs help setting up international roaming",
            "customer's voicemail isn't working",
            "customer has poor signal at home",
            "customer's new device won't activate",
            "customer needs help with WiFi calling setup",
            "customer is getting error messages on their phone",
            "customer needs troubleshooting for specific device issues"
        ]
    },
    
    "product": {
        "contexts": [
            QueryContext("product", True, False, "price-conscious customer", "plan comparison"),
            QueryContext("product", True, False, "tech enthusiast", "device upgrade"),
            QueryContext("product", True, False, "family customer", "multi-line planning"),
            QueryContext("product", True, False, "potential customer", "service exploration"), 
            QueryContext("product", True, False, "existing customer", "optimization"),
        ],
        "base_scenarios": [
            "customer wants to compare available plans",
            "customer is looking for device upgrade options",
            "customer needs information about current promotions",
            "customer wants to know about 5G compatibility",
            "customer is asking about plan features and benefits",
            "customer needs device specifications and pricing",
            "customer wants to understand family plan options",
            "customer is checking device trade-in values",
            "customer needs information about business plans",
            "customer wants to know about unlimited data options"
        ]
    },

    "multi_domain": {
        "contexts": [
            QueryContext("multi_domain", True, True, "frustrated customer", "complex billing and service issue"),
            QueryContext("multi_domain", True, False, "cost-conscious customer", "plan optimization with account review"),
            QueryContext("multi_domain", True, True, "business customer", "comprehensive service analysis"),
            QueryContext("multi_domain", True, False, "tech-savvy customer", "device and plan compatibility"),
            QueryContext("multi_domain", True, True, "family plan customer", "usage analysis and plan adjustment"),
            QueryContext("multi_domain", True, False, "new customer", "account setup with product questions"),
        ],
        "base_scenarios": [
            # Billing + Product scenarios
            "customer's bill is high and wants to explore plan downgrade options",
            "customer wants to understand usage charges and see if different plan would save money", 
            "customer is questioning overage fees and needs plan recommendations",
            "customer wants to compare their current plan costs with available alternatives",
            
            # Account + Billing scenarios  
            "customer can't access online account and needs to check payment status",
            "customer's autopay failed and wants to verify account settings",
            "customer moved and needs to update account info and understand billing changes",
            
            # Tech Support + Product scenarios
            "customer has connectivity issues and wants to know if device upgrade would help",
            "customer's device isn't working properly and wondering about replacement options",
            "customer needs help setting up new device and wants to know about compatible plans",
            
            # Tech Support + Account scenarios
            "customer can't receive calls and wants to verify their account is active",
            "customer has service issues and wants to check if account suspension is the cause",
            
            # Billing + Tech Support scenarios
            "customer is being charged for services that aren't working properly",
            "customer has high data charges and needs help understanding usage patterns",
            
            # Account + Product scenarios
            "customer wants to add a line and needs to understand how it affects their current plan",
            "customer's contract is ending and wants to review upgrade options",
            
            # Triple domain scenarios (Billing + Account + Product)
            "customer wants complete account review including usage, billing, and plan optimization",
            "customer is considering canceling and wants full service and cost analysis",
            "customer got married and needs to merge accounts, understand billing, and explore family plans",
            
            # All four domains
            "customer has service issues, billing disputes, wants account changes, and device upgrades",
        ]
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query Generator

# COMMAND ----------

class QueryGenerator:
    """Generate support queries using LLM endpoint."""
    
    def __init__(self, llm_endpoint: str = LLM_ENDPOINT):
        self.llm_endpoint = llm_endpoint
        self.deploy_client = deploy_client
        
    @retry(tries=3, delay=2)
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM endpoint with retry logic."""
        try:
            response = self.deploy_client.predict(
                endpoint=self.llm_endpoint,
                inputs={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise
    
    def generate_customer_id(self) -> str:
        """Generate random customer ID from the valid range."""
        customer_num = random.randint(CUSTOMER_ID_START, CUSTOMER_ID_END)
        return f"CUS-{customer_num:05d}"
    
    def generate_realistic_query(self, category: str, context: QueryContext, scenario: str) -> str:
        """Generate realistic customer service query."""
        
        system_prompt = f"""You are a human customer service agent helping customers with telecom support.
You need to phrase customer questions naturally, as if you're asking on behalf of the customer.

Context: You're dealing with a {context.persona_context} who has a {context.business_scenario}.
Category: {category}
Scenario: {scenario}

Generate a natural, conversational question that a customer service agent would ask.
The question should:
- Sound like something a real customer would ask
- Be specific and actionable
- Include relevant details that make it realistic
- Be appropriate for the {category} category
- Reflect the persona of a {context.persona_context}

Examples of good questions:
- Customer is asking what plan they're currently on and when it expires
- Customer sees a $25 charge on their April bill they don't recognize - can you explain what it's for?
- The customer's iPhone 15 isn't getting 5G speeds even though they're in a coverage area
- Are there any promotions for upgrading to unlimited data
- How much data did the customer use last month?
- Customer wants to know their voice minutes usage for May 2025
- Customer is asking about their total usage breakdown for the last billing cycle

Generate ONE realistic question following this pattern. Don't include any preamble or explanation."""

        user_prompt = f"Generate a realistic {category} query for: {scenario}"
        
        return self._call_llm(system_prompt, user_prompt)
    
    def generate_temporal_context(self) -> Dict[str, Any]:
        """Generate date contexts for billing queries."""
        now = datetime.now()
        
        contexts = [
            {
                "description": "current month",
                "start_date": now.replace(day=1).strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d")
            },
            {
                "description": "last month",
                "start_date": (now.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d"),
                "end_date": (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
            },
            {
                "description": "last 3 months",
                "start_date": (now.replace(day=1) - timedelta(days=90)).strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d")
            },
            {
                "description": "last billing cycle",
                "start_date": (now.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d"),
                "end_date": (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
            },
            {
                "description": "past 30 days",
                "start_date": (now - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d")
            },
            {
                "description": "past 7 days", 
                "start_date": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d")
            },
            {
                "description": "April 2025",
                "start_date": "2025-04-01",
                "end_date": "2025-04-30"
            },
            {
                "description": "May 2025",
                "start_date": "2025-05-01",
                "end_date": "2025-05-31"
            }
        ]
        
        return random.choice(contexts)
    
    def enhance_query_with_context(self, query: str, context: QueryContext) -> Tuple[str, Dict[str, Any]]:
        """Enhance query with appropriate context (customer ID, dates, etc.)."""
        custom_inputs = {}
        
        # always add customer ID
        custom_inputs["customer"] = self.generate_customer_id()
        
        # add temporal context for billing queries
        if context.requires_dates and "billing" in context.category:
            temporal_ctx = self.generate_temporal_context()
            # inject dates into the query if not already present
            if not any(term in query.lower() for term in ["april", "may", "june", "month", "2025"]):
                time_phrase = temporal_ctx["description"]
                query = query.replace("bill", f"{time_phrase} bill")
        
        return query, custom_inputs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Client

# COMMAND ----------

class TelcoAgentClient:
    """Client for interacting with telco support agent endpoint."""
    
    def __init__(self, endpoint_name: str = AGENT_ENDPOINT_NAME):
        self.endpoint_name = endpoint_name
        self.deploy_client = deploy_client
        
    @retry(tries=3, delay=2, backoff=2)
    def query_agent(self, query: str, custom_inputs: Dict[str, Any] = None) -> Tuple[Dict[str, Any], str]:
        """Send query to the agent endpoint and return response with trace ID.
        
        Returns:
            Tuple of (response, trace_id)
        """
        
        payload = {
            "input": [{"role": "user", "content": query}],
            "databricks_options": {
                "return_trace": True
            }
        }
        
        if custom_inputs:
            payload["custom_inputs"] = custom_inputs
            
        try:
            response = self.deploy_client.predict(
                endpoint=self.endpoint_name,
                inputs=payload
            )
            
            trace_id = None
            if isinstance(response, dict):
                databricks_output = response.get('databricks_output', {})
                if isinstance(databricks_output, dict):
                    trace_info = databricks_output.get('trace', {})
                    if isinstance(trace_info, dict):
                        info = trace_info.get('info', {})
                        if isinstance(info, dict):
                            trace_id = info.get('trace_id')
                
                if not trace_id and isinstance(databricks_output, dict):
                    trace_id = (databricks_output.get('trace_id') or
                              databricks_output.get('request_id'))
                
                if not trace_id:
                    trace_id = response.get('trace_id')
            
            return response, trace_id
            
        except Exception as e:
            print(f"Request failed: {e}")
            raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feedback Generator

# COMMAND ----------

from mlflow.genai import judges

class FeedbackGenerator:
    """Generate synthetic feedback for agent responses."""
    
    def __init__(self):
        self.agent_names = AGENT_NAMES.copy()
        random.shuffle(self.agent_names)  # randomize order
        
    def get_random_agent(self) -> str:
        """Get a random agent name."""
        return random.choice(self.agent_names)
    
    def generate_feedback(self, query: str, response: Dict[str, Any], metadata: Dict[str, Any], trace_id: str) -> List[Dict[str, Any]]:
        """Generate varied feedback for a query/response pair."""
        feedbacks = []
        agent_name = self.get_random_agent()
        
        # Generate 1-3 different feedback items per query
        num_feedbacks = random.randint(1, 4)
        
        feedback_types = [
            self._generate_user_feedback,
            self._generate_query_answered,
            self._generate_user_router_feedback
            # self._generate_accuracy_feedback,
            # self._generate_clarity_feedback,
            # self._generate_completeness_feedback,
            # self._generate_relevance_feedback,
            # self._generate_data_usage_feedback,
        ]
        
        selected_feedback_types = random.sample(feedback_types, min(num_feedbacks, len(feedback_types)))
        
        for feedback_func in selected_feedback_types:
            feedback = feedback_func(query, response, metadata, agent_name, trace_id)
            if feedback:
                feedbacks.append(feedback)
        
        return feedbacks
    
    def _generate_user_feedback(self, query: str, response: Dict[str, Any], 
                                metadata: Dict[str, Any], agent_name: str, trace_id: str) -> Dict[str, Any]:
        """Generate user feedback by reading the response."""
        text_response = response['output'][-1]['content'][-1]['text']
        user_feedback = judges.meets_guidelines(
            guidelines=[
                "The response must not be an error message, it must sound like a person would respond.",
                "The response must not say that there were any technical errors or issues."
            ],
            context={
                "response": text_response
            }
        )
        # Don't log good feedback as this is not realistic.
        if user_feedback.value == "yes" and random.random() < 0.8:
            return
        
        real_route, expected_route = self._compute_correct_routing(
            query, response, metadata, agent_name, trace_id
        )

        # Don't log all the time as this is not realistic.
        should_log = random.random() < 0.5 or real_route != expected_route
        if not should_log:
            return
        
        human_rationale = self._call_llm(
            system_prompt="""
                Your job is to convert computer generated explanations about why a certain response was bad or good into a human readable explanation.
                The response should be short and concise as if a human explained why a response was bad after they gaves a thumbs down. Usually should be less than 10 words and can sometimes be formal, sometimes informal.

                As input you will get:
                <response>The agent's response</response>
                <computer_rationale>A computer generated rationale of why the response was bad.</computer_rationale>

                For example:
                <response>Max iterations (10) reached. Stopping.</response>
                <computer_rationale>The response 'Max iterations (10) reached. Stopping.' is an error message and indicates a technical issue. Therefore, it does not satisfy the guideline that the response must not be an error message and must not say that there were any technical errors or issues.<computer_rationale>

                Example outputs:
                - the agent seems broken
                - help me
                - the agent is broken
                - the agent is not working
            """,
            user_prompt=f"""
            <response>{text_response}</response>
            <computer_rationale>{user_feedback.rationale}</computer_rationale>
            """
        )

        return {
            "name": "user_feedback",
            "value": user_feedback.value == "yes",
            "source": agent_name,
            "rationale": human_rationale
        }

    @retry(tries=3, delay=2)
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM endpoint with retry logic."""
        try:
            response = deploy_client.predict(
                endpoint=LLM_ENDPOINT,
                inputs={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise
    
    def _generate_user_router_feedback(self, query: str, response: Dict[str, Any], 
                                metadata: Dict[str, Any], agent_name: str, trace_id: str) -> Dict[str, Any]:
        route_output, good_supervisor_response = self._compute_correct_routing(query, response, metadata, agent_name, trace_id)
        if route_output != good_supervisor_response:
            return {
                "name": "user_feedback",
                "value": False,
                "source": "databricks",
                "rationale": f"Route: {route_output}, Best route: {good_supervisor_response}"
            }
            
    def _compute_correct_routing(self, query: str, response: Dict[str, Any], 
                                metadata: Dict[str, Any], agent_name: str, trace_id: str) -> Dict[str, Any]:
        """Generate user feedback by reading the response."""
        text_response = response['output'][-1]['content'][-1]['text']
        good_supervisor_prompt = """
            You are an intelligent AI assistant for a telecom customer support system. Your job is to analyze customer queries and route them to the appropriate specialized agent based on the query's intent, required data sources, and the specific capabilities of each agent.

            ## SPECIALIZED AGENTS AND THEIR CAPABILITIES:

            ### 1. ACCOUNT AGENT
            **Purpose**: Customer profile information, account status, subscription details, and account management
            **Data Access**: Customer profiles, subscription details, plan information, account metrics 
            **Route to ACCOUNT for queries about**:
            - Current plan/subscription details ("What plan am I on?", "Show me my subscriptions")
            - Account status and profile ("Is my account active?", "What's my loyalty tier?")
            - Account history ("When did I create my account?", "How long have I been a customer?")
            - Subscription management ("Is autopay enabled on my account?", "How many lines do I have?")
            - Customer segment information ("What customer tier am I in?")

            ### 2. BILLING AGENT  
            **Purpose**: Bills, payments, charges, billing cycles, usage data, and financial inquiries
            **Data Access**: Billing records, payment history, usage statistics, billing cycles
            **Route to BILLING for queries about**:
            - Bill amounts and charges ("Why is my bill $X?", "What are my charges for [date]?")
            - Payment information ("When is my payment due?", "Did my payment go through?")
            - Usage statistics ("How much data did I use?", "Show me my usage for [period]")
            - Billing disputes ("I don't recognize this charge", "My bill seems wrong")
            - Billing cycles and patterns ("Why is my bill different this month?")

            ### 3. TECH_SUPPORT AGENT
            **Purpose**: Technical troubleshooting, connectivity issues, device problems, and how-to guidance
            **Data Access**: Knowledge base articles (FAQs, troubleshooting guides), historical support tickets
            **Tools Available**:
            **Route to TECH_SUPPORT for queries about**:
            - Device connectivity ("My phone won't connect", "WiFi not working")
            - Service issues ("Can't make calls", "No signal", "Internet is slow")
            - Technical setup ("How do I set up voicemail?", "Configure international roaming")
            - Troubleshooting steps ("My device keeps restarting", "Apps won't work")
            - Technical how-to questions ("Reset network settings", "Update device software")

            ### 4. PRODUCT AGENT
            **Purpose**: Service plans, devices, promotions, product comparisons, and recommendations
            **Data Access**: Plan catalog, device specifications, current promotions, customer device info
            **Route to PRODUCT for queries about**:
            - Plan comparisons ("What's the difference between Standard and Premium?")
            - Device information ("Is my phone 5G compatible?", "What Samsung devices are available?")
            - Promotions and offers ("Do you have any promotions?", "Are there iPhone upgrade deals?")
            - Product recommendations ("Which plan has the most data?", "Best device for my needs?")
            - Upgrade eligibility ("Can I upgrade my device early?", "What plans work with 5G?")

            ## ROUTING DECISION CRITERIA:

            **Primary Indicators**:
            - Keywords related to money/payments → BILLING
            - Technical problems/troubleshooting → TECH_SUPPORT  
            - Plan/device comparisons or shopping → PRODUCT
            - Account status/profile questions → ACCOUNT

            **For Ambiguous Queries**:
            - "What devices do I have?" → PRODUCT (customer-specific device info)
            - "When does my contract end?" → ACCOUNT (subscription details)
            - "How much do I owe?" → BILLING (payment/balance info)
            - "My service isn't working" → TECH_SUPPORT (troubleshooting needed)

            **Multi-Domain Queries**:
            - If query spans multiple domains, route to the PRIMARY domain:
                - "How much does the Premium plan cost and do I qualify?" → PRODUCT (primary: plan info)
                - "My bill is high, what plan options do I have?" → BILLING (primary: bill inquiry)

            ## RESPONSE FORMAT:
            Analyze the query and respond with ONLY ONE of these four agent types:
            - account
            - billing  
            - tech_support
            - product

            Do NOT include any explanation, reasoning, or additional text in your response.
        """

        good_supervisor_response = self._call_llm(
            system_prompt=good_supervisor_prompt,
            user_prompt=f"""<request>{query}</query>"""
        )
        trace = mlflow.get_trace(trace_id)
        route_query_span = trace.search_spans(name='route_query')
        route_output = route_query_span[0].outputs

        print('Route comparison, bad output:', route_output, 'better route:', good_supervisor_response)

        return route_output, good_supervisor_response

    def _generate_query_answered(self, query: str, response: Dict[str, Any], 
                                metadata: Dict[str, Any], agent_name: str, trace_id: str) -> Dict[str, Any]:
        """Generate user feedback by reading the response."""
        text_response = response['output'][-1]['content'][-1]['text']
        query_answered = judges.meets_guidelines(
            guidelines=[
                "The response answer the user's query.",
                "The response must be helpful."
            ],
            context={
                "response": text_response
            }
        )

        return {
            "name": "query_answered",
            "value": query_answered.value,
            "source": "databricks",
            "source_type": AssessmentSourceType.LLM_JUDGE,
            "rationale": query_answered.rationale
        }

    def _generate_helpfulness_feedback(self, query: str, response: Dict[str, Any], 
                                     metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate helpfulness feedback (mostly positive)."""
        # 85% positive feedback
        is_helpful = random.random() < 0.85
        
        return {
            "name": "helpfulness",
            "value": is_helpful,
            "source": agent_name,
            "rationale": random.choice([
                "Response directly addressed the customer's question with actionable information",
                "Customer seemed satisfied with the level of detail provided",
                "The agent provided clear next steps for the customer",
                "Response was comprehensive and covered all aspects of the inquiry"
            ]) if is_helpful else random.choice([
                "Customer needed additional clarification after the response",
                "Response could have been more specific to the customer's situation",
                "Some important details were missing from the response"
            ])
        }
    
    def _generate_accuracy_feedback(self, query: str, response: Dict[str, Any], 
                                  metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate accuracy feedback."""
        # 90% accurate feedback
        is_accurate = random.random() < 0.90
        
        return {
            "name": "accuracy",
            "value": is_accurate,
            "source": agent_name,
            "rationale": random.choice([
                "Information provided was consistent with company policies",
                "Data retrieved appeared correct based on customer account",
                "Technical guidance was sound and appropriate",
                "Billing information matched customer records"
            ]) if is_accurate else random.choice([
                "Minor discrepancy noted in billing calculation",
                "One piece of information needed verification",
                "Policy reference could be more current"
            ])
        }
    
    def _generate_clarity_feedback(self, query: str, response: Dict[str, Any], 
                                 metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate clarity feedback."""
        # 80% clear feedback
        is_clear = random.random() < 0.80
        
        return {
            "name": "clarity",
            "value": is_clear,
            "source": agent_name,
            "rationale": random.choice([
                "Response was easy to understand and well-structured",
                "Technical terms were explained appropriately",
                "Step-by-step instructions were clear and logical",
                "Customer could easily follow the guidance provided"
            ]) if is_clear else random.choice([
                "Some technical language could be simplified",
                "Response structure could be more organized",
                "Customer asked for clarification on certain points"
            ])
        }
    
    def _generate_completeness_feedback(self, query: str, response: Dict[str, Any], 
                                      metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate completeness feedback."""
        # 80% complete feedback
        is_complete = random.random() < 0.8
        
        return {
            "name": "completeness",
            "value": is_complete,
            "source": agent_name,
            "rationale": random.choice([
                "All aspects of the customer's question were addressed",
                "Response included relevant additional information",
                "Follow-up options were clearly provided",
                "No further questions were needed from the customer"
            ]) if is_complete else random.choice([
                "Customer had follow-up questions about specific details",
                "One aspect of the inquiry could have been expanded",
                "Additional context would have been helpful"
            ])
        }
    
    def _generate_relevance_feedback(self, query: str, response: Dict[str, Any], 
                                   metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate relevance feedback for telco-specific queries."""
        # 85% relevant feedback
        is_relevant = random.random() < 0.85
        
        category = metadata.get("category", "unknown")
        
        return {
            "name": "relevance_to_query",
            "value": is_relevant,
            "source": agent_name,
            "rationale": random.choice([
                f"Response addressed the {category} query with appropriate telco-specific information",
                f"Answer was well-suited to the customer's {category} needs",
                f"Response included relevant {category} details and next steps",
                f"Content was appropriate for a {category} support interaction"
            ]) if is_relevant else random.choice([
                f"Response could have been more specific to {category} domain",
                f"Some {category} context was missing from the response",
                f"Answer was too generic for this {category} query"
            ])
        }
    
    def _generate_data_usage_feedback(self, query: str, response: Dict[str, Any], 
                                    metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate feedback on proper use of customer data."""
        # 90% proper data usage
        proper_data_use = random.random() < 0.90
        
        return {
            "name": "proper_data_usage",
            "value": proper_data_use,
            "source": agent_name,
            "rationale": random.choice([
                "Agent appropriately used customer-specific data in response",
                "Response referenced correct customer account information",
                "Customer data was used securely and appropriately",
                "Agent accessed only necessary customer information"
            ]) if proper_data_use else random.choice([
                "Could have used more specific customer data in response",
                "Response was too generic given available customer context",
                "Some relevant customer information was not utilized"
            ])
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feedback Logger

# COMMAND ----------

class FeedbackLogger:
    """Handle feedback logging."""
    
    @staticmethod
    def log_feedback_for_query(feedbacks: List[Dict[str, Any]], trace_id: str) -> bool:
        """Log feedback items.
        
        Returns:
            True if all feedback was logged successfully, False otherwise
        """
        if not trace_id:
            print("❌ No trace ID provided for feedback logging")
            return False
        
        success_count = 0
        total_count = len(feedbacks)
        
        print(f"Logging {total_count} feedback items to trace {trace_id}")
        
        time.sleep(2)
        for feedback in feedbacks:
            try:
                feedback_name = feedback["name"]
                feedback_value = feedback["value"]
                feedback_source = feedback["source"]
                feedback_rationale = feedback["rationale"]
                feedback_source_type = feedback.get("source_type", AssessmentSourceType.HUMAN)
                source = AssessmentSource(
                    source_type=feedback_source_type,
                    source_id=feedback_source,
                )
                
                mlflow.log_feedback(
                    trace_id=trace_id,
                    name=feedback_name,
                    source=source,
                    value=feedback_value,
                    rationale=feedback_rationale,
                )
                
                success_count += 1
                print(f"  ✅ {feedback_name}: {feedback_value} (by {feedback_source})")
                    
            except Exception as e:
                print(f"  ❌ Failed to log {feedback.get('name', 'unknown')}: {e}")
                
                if "Invalid request id" in str(e) or "Request id must map to a trace" in str(e):
                    try:
                        print(f"    Retrying feedback logging after longer delay...")
                        time.sleep(3)
                        
                        mlflow.log_feedback(
                            trace_id=trace_id,
                            name=feedback_name,
                            source=source,
                            value=feedback_value,
                            rationale=feedback_rationale,
                        )
                        
                        success_count += 1
                        print(f"  ✅ {feedback_name}: {feedback_value} (by {feedback_source}) [retry successful]")
                        
                    except Exception as retry_e:
                        print(f"    ❌ Retry also failed: {retry_e}")
        
        print(f"Feedback logging complete: {success_count}/{total_count} successful")
        return success_count == total_count

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic Query Engine

# COMMAND ----------

class SyntheticQueryEngine:
    """Generate and execute synthetic queries."""
    
    def __init__(self, num_queries: int = QUERIES_PER_BATCH, include_multi_domain: bool = INCLUDE_MULTI_DOMAIN):
        self.num_queries = num_queries
        self.include_multi_domain = include_multi_domain
        self.generator = QueryGenerator()
        self.client = TelcoAgentClient()
        self.feedback_generator = FeedbackGenerator()
        self.feedback_logger = FeedbackLogger()
        self.formatter = ResponseFormatter()
        
    def generate_query_batch(self) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Generate a batch of diverse queries."""
        queries = []
        
        all_categories = list(QUERY_TEMPLATES.keys())
        
        if self.include_multi_domain:
            categories = all_categories
            # 80% single domain, 20% multi domain
            single_domain_queries = int(self.num_queries * 0.8)
            multi_domain_queries = self.num_queries - single_domain_queries
            print(f"Generating {single_domain_queries} single-domain and {multi_domain_queries} multi-domain queries")
        else:
            categories = [cat for cat in all_categories if cat != "multi_domain"]
            single_domain_queries = self.num_queries
            multi_domain_queries = 0
            print(f"Generating {single_domain_queries} single-domain queries only")
        
        # single domain queries
        single_domain_categories = [cat for cat in categories if cat != "multi_domain"]
        queries_per_single_category = single_domain_queries // len(single_domain_categories)
        extra_queries = single_domain_queries % len(single_domain_categories)
        
        for i, category in enumerate(single_domain_categories):
            # calculate number of queries for category
            category_query_count = queries_per_single_category
            if i < extra_queries:
                category_query_count += 1
                
            # generate queries for this category
            contexts = QUERY_TEMPLATES[category]["contexts"]
            scenarios = QUERY_TEMPLATES[category]["base_scenarios"]
            
            for _ in range(category_query_count):
                context = random.choice(contexts)
                scenario = random.choice(scenarios)
                
                try:
                    # generate query
                    query = self.generator.generate_realistic_query(category, context, scenario)
                    
                    # add appropriate context (customer ID, dates, etc.)
                    enhanced_query, custom_inputs = self.generator.enhance_query_with_context(query, context)
                    
                    # create metadata for tracking
                    metadata = {
                        "category": category,
                        "persona": context.persona_context,
                        "scenario": context.business_scenario,
                        "requires_customer_id": context.requires_customer_id,
                        "requires_dates": context.requires_dates,
                        "customer_id": custom_inputs.get("customer"),
                        "generated_at": datetime.now().isoformat()
                    }
                    
                    queries.append((enhanced_query, custom_inputs, metadata))
                    
                except Exception as e:
                    print(f"Failed to generate query for {category}: {e}")
                    continue
        
        # multi-domain queries ONLY if flag is enabled
        if self.include_multi_domain and multi_domain_queries > 0:
            print(f"Generating {multi_domain_queries} multi-domain queries...")
            contexts = QUERY_TEMPLATES["multi_domain"]["contexts"]
            scenarios = QUERY_TEMPLATES["multi_domain"]["base_scenarios"]
            
            for _ in range(multi_domain_queries):
                context = random.choice(contexts)
                scenario = random.choice(scenarios)
                
                try:
                    # generate multi-domain query
                    query = self.generator.generate_realistic_query("multi_domain", context, scenario)
                    
                    # add appropriate context (customer ID, dates, etc.)
                    enhanced_query, custom_inputs = self.generator.enhance_query_with_context(query, context)
                    
                    # create metadata for tracking with expected domains
                    metadata = {
                        "category": "multi_domain",
                        "persona": context.persona_context,
                        "scenario": context.business_scenario,
                        "requires_customer_id": context.requires_customer_id,
                        "requires_dates": context.requires_dates,
                        "customer_id": custom_inputs.get("customer"),
                        "generated_at": datetime.now().isoformat(),
                        "expected_domains": self._extract_expected_domains(scenario)
                    }
                    
                    queries.append((enhanced_query, custom_inputs, metadata))
                    
                except Exception as e:
                    print(f"Failed to generate multi-domain query: {e}")
                    continue
        
        random.shuffle(queries)
        return queries

    def _extract_expected_domains(self, scenario: str) -> List[str]:
        """Extract which domains a multi-domain scenario should ideally involve."""
        domain_keywords = {
            "billing": ["bill", "payment", "charge", "cost", "usage", "overage", "fee", "billing"],
            "account": ["account", "autopay", "subscription", "contract", "line", "profile", "login", "access"],
            "product": ["plan", "device", "upgrade", "option", "alternative", "family", "business", "downgrade"],
            "tech_support": ["connectivity", "working", "service", "setup", "issue", "problem", "technical", "connection"]
        }
        
        expected_domains = []
        scenario_lower = scenario.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in scenario_lower for keyword in keywords):
                expected_domains.append(domain)
        
        return expected_domains

    def execute_query_batch(self, queries: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
                           max_workers: int = MAX_WORKERS) -> List[Dict[str, Any]]:
        """Execute batch of queries with parallel processing."""
        
        results = []
        
        def execute_single_query(query_data):
            query, custom_inputs, metadata = query_data
            start_time = time.time()
            
            try:
                print(f"🔍 Executing: {query}...")
                
                # Get response and trace ID
                response, trace_id = self.client.query_agent(query, custom_inputs)
                execution_time = time.time() - start_time
                
                # Generate feedback
                feedbacks = self.feedback_generator.generate_feedback(query, response, metadata, trace_id)
                
                # Log feedback
                feedback_success = False
                if trace_id:
                    feedback_success = self.feedback_logger.log_feedback_for_query(feedbacks, trace_id)
                else:
                    print("⚠️  No trace ID available - skipping feedback logging")
                
                return {
                    "query": query,
                    "custom_inputs": custom_inputs,
                    "metadata": metadata,
                    "response": response,
                    "execution_time": execution_time,
                    "feedbacks": feedbacks,
                    "feedback_logged": feedback_success,
                    "success": True,
                    "error": None,
                    "trace_id": trace_id
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                print(f"❌ Query failed: {error_msg}")
                
                return {
                    "query": query,
                    "custom_inputs": custom_inputs,
                    "metadata": metadata,
                    "response": None,
                    "execution_time": execution_time,
                    "feedbacks": [],
                    "feedback_logged": False,
                    "success": False,
                    "error": error_msg,
                    "trace_id": None
                }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(execute_single_query, query_data): query_data 
                             for query_data in queries}
            
            for future in as_completed(future_to_query):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Future execution failed: {e}")
        
        return results
    
    def log_batch_summary(self, results: List[Dict[str, Any]]):
        """Log summary statistics for the batch."""
        total_queries = len(results)
        successful_queries = sum(1 for r in results if r["success"])
        failed_queries = total_queries - successful_queries
        feedback_logged = sum(1 for r in results if r.get("feedback_logged", False))
        
        if successful_queries > 0:
            avg_execution_time = sum(r["execution_time"] for r in results if r["success"]) / successful_queries
        else:
            avg_execution_time = 0
        
        category_stats = {}
        for result in results:
            category = result["metadata"]["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0}
            category_stats[category]["total"] += 1
            if result["success"]:
                category_stats[category]["success"] += 1
        
        print(f"\n{'='*60}")
        print(f"BATCH EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Queries: {total_queries}")
        print(f"Successful: {successful_queries}")
        print(f"Failed: {failed_queries}")
        print(f"Success Rate: {(successful_queries/total_queries)*100:.1f}%")
        print(f"Average Execution Time: {avg_execution_time:.2f}s")
        print(f"Feedback Logged: {feedback_logged}/{successful_queries}")
        
        print(f"\nCategory Breakdown:")
        for category, stats in category_stats.items():
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"  {category.title()}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # feedback summary
        total_feedbacks = sum(len(r["feedbacks"]) for r in results if r["success"])
        print(f"\nFeedback Generated: {total_feedbacks} items")
        
        if failed_queries > 0:
            print(f"\nFailed Queries:")
            for result in results:
                if not result["success"]:
                    print(f"  - {result['query'][:80]}... | Error: {result['error']}")
        
        print(f"{'='*60}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Functions

# COMMAND ----------

def generate_sample_queries_for_testing():    
    engine = SyntheticQueryEngine(num_queries=5)
    queries = engine.generate_query_batch()
    
    print("SAMPLE QUERIES FOR TESTING")
    print("=" * 50)
    
    for i, (query, custom_inputs, metadata) in enumerate(queries):
        print(f"{i+1}. Category: {metadata['category'].upper()}")
        print(f"   Persona: {metadata['persona']}")
        print(f"   Query: {query}")
        if custom_inputs:
            print(f"   Custom Inputs: {custom_inputs}")
        print(f"   Request JSON:")
        request = {
            "input": [{"role": "user", "content": query}],
            "databricks_options": {"return_trace": True}
        }
        if custom_inputs:
            request["custom_inputs"] = custom_inputs
        print(f"   {json.dumps(request, indent=2)}")
        print()

def test_single_query():
    print("MANUAL SINGLE QUERY TEST")
    print("=" * 40)
    
    # create components
    generator = QueryGenerator()
    client = TelcoAgentClient()
    feedback_gen = FeedbackGenerator()
    feedback_logger = FeedbackLogger()
    formatter = ResponseFormatter()
    
    # generate a single query
    test_query = "Customer wants to know what plan they're currently on and when their contract expires"
    test_custom_inputs = {"customer": generator.generate_customer_id()}
    
    print(f"Test query: {test_query}")
    print(f"Custom inputs: {test_custom_inputs}")
    
    try:
        # execute query
        print(f"\nExecuting query...")
        start_time = time.time()
        response, trace_id = client.query_agent(test_query, test_custom_inputs)
        execution_time = time.time() - start_time
        
        print(f"✅ Query completed successfully!")
        print(f"Trace ID: {trace_id}")
        
        # Format and display response summary
        print(f"\nRESPONSE SUMMARY:")
        print(f"{formatter.format_response_summary(response, execution_time)}")
        
        # Generate and log feedback
        if trace_id:
            print(f"\nGenerating and logging feedback...")
            metadata = {"category": "account", "persona": "test", "scenario": "test"}
            feedbacks = feedback_gen.generate_feedback(test_query, response, metadata, trace_id)
            feedback_success = feedback_logger.log_feedback_for_query(feedbacks, trace_id)
            
            if feedback_success:
                print(f"✅ All feedback logged successfully!")
            else:
                print(f"⚠️  Some feedback logging failed")
        else:
            print(f"⚠️  No trace ID - cannot log feedback")
        
        print(f"\n✅ Single query test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Single query test failed: {e}")
        return False

def test_small_batch():
    print("TESTING WITH SMALL BATCH")
    print("=" * 50)
    
    # create a small test engine
    test_engine = SyntheticQueryEngine(num_queries=5)
    formatter = ResponseFormatter()
    
    # generate small batch
    print("🔧 Generating test queries...")
    queries = test_engine.generate_query_batch()
    
    print(f"Generated {len(queries)} test queries:")
    for i, (query, custom_inputs, metadata) in enumerate(queries):
        print(f"\n{i+1}. [{metadata['category'].upper()}] {query}")
        if custom_inputs:
            print(f"    Custom inputs: {custom_inputs}")
    
    # execute the batch
    print(f"\nExecuting test batch...")
    results = test_engine.execute_query_batch(queries, max_workers=2)
    
    # show results
    test_engine.log_batch_summary(results)
    
    # show detailed results for successful queries
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(successful_results[:2]):  # show first 2
            print(f"\n{'='*50}")
            print(f"Query {i+1}: {result['query']}")
            print(f"Trace ID: {result.get('trace_id', 'N/A')}")
            print(f"Feedbacks: {len(result['feedbacks'])} generated, logged: {result.get('feedback_logged', False)}")
            
            if result['response']:
                print(f"\n{formatter.format_response_summary(result['response'], result['execution_time'])}")
            
            if result['feedbacks']:
                print(f"\nFeedback Details:")
                for feedback in result['feedbacks']:
                    status = "✅" if feedback['value'] else "❌"
                    print(f"  {status} {feedback['name']}: {feedback['value']} (by {feedback['source']})")
                    print(f"    {feedback['rationale']}")
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Execution Functions

# COMMAND ----------

def run_synthetic_query_batch(num_queries: int = QUERIES_PER_BATCH) -> Dict[str, Any]:
    """Run a full batch of synthetic queries."""
    print(f"RUNNING SYNTHETIC QUERY BATCH ({num_queries} queries)")
    print("=" * 60)
    
    batch_start_time = time.time()
        
    # create engine
    engine = SyntheticQueryEngine(num_queries=num_queries)
    
    # generate queries
    print("Generating query batch...")
    queries = engine.generate_query_batch()
    print(f"Generated {len(queries)} queries")
    
    # execute batch
    print("Executing query batch...")
    results = engine.execute_query_batch(queries)
    
    # calculate total time
    batch_total_time = time.time() - batch_start_time
    
    # log batch summary
    engine.log_batch_summary(results)
    
    # create summary
    summary = {
        "total_queries": len(queries),
        "successful_queries": sum(1 for r in results if r["success"]),
        "failed_queries": sum(1 for r in results if not r["success"]),
        "feedback_logged": sum(1 for r in results if r.get("feedback_logged", False)),
        "total_execution_time": batch_total_time,
        "total_feedbacks": sum(len(r["feedbacks"]) for r in results if r["success"]),
        "results": results
    }
    
    print(f"✅ Batch execution completed in {batch_total_time:.2f}s")
    return summary

def run_continuous_simulation(batches: int = 3, delay_between_batches: int = 300):
    """Run continuous simulation with multiple batches."""
    print(f"STARTING CONTINUOUS SIMULATION ({batches} batches)")
    print("=" * 60)
    
    simulation_start_time = time.time()
    all_summaries = []
    
    for batch_num in range(1, batches + 1):
        print(f"\nStarting batch {batch_num}/{batches}")
        
        try:
            summary = run_synthetic_query_batch()
            all_summaries.append(summary)
            
            print(f"✅ Batch {batch_num} completed successfully")
            
            # delay between batches (except for the last one)
            if batch_num < batches:
                print(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
                
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            continue
        
    simulation_total_time = time.time() - simulation_start_time
    
    # log simulation summary
    total_queries = sum(s["total_queries"] for s in all_summaries)
    total_successful = sum(s["successful_queries"] for s in all_summaries)
    total_feedbacks = sum(s["total_feedbacks"] for s in all_summaries)
    total_feedback_logged = sum(s["feedback_logged"] for s in all_summaries)
    
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total batches: {len(all_summaries)}")
    print(f"Total queries: {total_queries}")
    print(f"Total successful: {total_successful}")
    print(f"Total feedbacks: {total_feedbacks}")
    print(f"Feedback logged: {total_feedback_logged}")
    print(f"Total time: {simulation_total_time:.2f}s")
    print(f"{'='*60}")
    
    return {
        "batches_completed": len(all_summaries),
        "total_simulation_time": simulation_total_time,
        "batch_summaries": all_summaries
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Execution

# COMMAND ----------

# print("Generating sample queries...")
generate_sample_queries_for_testing()

# COMMAND ----------

# print("Running single query test...")
single_test_success = test_single_query()

# COMMAND ----------

# print("Running small batch test...")
test_results = test_small_batch()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Execution
# MAGIC
# MAGIC Uncomment cell to run full synthetic query batch.

# COMMAND ----------

# print("Running batch execution...")
# batch_summary = run_synthetic_query_batch(num_queries=QUERIES_PER_BATCH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Continuous Simulation
# MAGIC
# MAGIC Uncomment to run continuous simulation (multiple batches with delays)

# COMMAND ----------

print("Running continuous simulation...")
simulation_summary = run_continuous_simulation(batches=6, delay_between_batches=120)
print(f"Simulation summary: {simulation_summary}")

# COMMAND ----------

# import pyspark.sql.functions as F
# spark.table("telco_customer_support_dev.agent.telco_customer_support_agent_1_payload").orderBy(F.col("request_time").desc()).display()