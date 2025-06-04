# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Query Generator
# MAGIC 
# MAGIC Notebook to generate synthetic queries to test telco support agent endpoint.
# MAGIC Can be scheduled as a Databricks job to continuously simulate customer interactions.

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q
# MAGIC %pip install mlflow databricks-agents --upgrade --pre
# MAGIC %pip install retry textstat
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import mlflow
from retry import retry
from mlflow.deployments import get_deploy_client
from databricks.agents.evals import judges
from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType
import textstat

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if root_path:
    sys.path.append(root_path)

from telco_support_agent.data.config import CONFIG

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

# Config
AGENT_ENDPOINT_NAME = "telco-customer-support-agent"
LLM_ENDPOINT = "databricks-claude-3-7-sonnet"
MAX_WORKERS = 5  # parallel query execution
QUERIES_PER_BATCH = 50  # number of queries to generate per execution

# Customer ID generation based on data generation config
CUSTOMER_ID_START = 10001
CUSTOMER_COUNT = CONFIG["volumes"]["customers"]  # 1500 customers
CUSTOMER_ID_END = CUSTOMER_ID_START + CUSTOMER_COUNT - 1

# generated human agents for feedback capture
AGENT_NAMES = [
    "Sarah Chen", "Michael Rodriguez", "Jessica Williams", "David Park",
    "Emily Johnson", "Robert Taylor", "Amanda Davis", "James Wilson",
    "Maria Garcia", "Christopher Lee", "Ashley Brown", "Daniel Kim",
    "Lisa Anderson", "Kevin Martinez", "Rachel Thompson", "Brandon White"
]

print(f"Customer ID range: CUS-{CUSTOMER_ID_START:05d} to CUS-{CUSTOMER_ID_END:05d}")
print(f"Total possible customers: {CUSTOMER_COUNT}")

deploy_client = get_deploy_client("databricks")

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

# Query generation templates organized by agent category
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
            QueryContext("billing", True, True, "data-heavy user", "usage verification"),
            QueryContext("billing", True, True, "traveling customer", "roaming charges"),
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
            "customer is planning data usage for upcoming month"
        ]
    },
    
    "tech_support": {
        "contexts": [
            QueryContext("tech_support", False, False, "frustrated customer", "service disruption"),
            QueryContext("tech_support", False, False, "tech-savvy customer", "configuration issue"),
            QueryContext("tech_support", False, False, "elderly customer", "device setup help"),
            QueryContext("tech_support", False, False, "business customer", "connectivity problem"),
            QueryContext("tech_support", False, False, "traveling customer", "roaming setup"),
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
            QueryContext("product", False, False, "potential customer", "service exploration"),
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
        """Call LLM with retry logic."""
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
- "Customer is asking what plan they're currently on and when it expires"
- "Customer sees a $25 charge on their April bill they don't recognize - can you explain what it's for?"
- "Customer's iPhone 15 isn't getting 5G speeds even though they're in a coverage area"
- "Customer wants to know if there are any promotions for upgrading to unlimited data"

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
        
        # add customer ID if required
        if context.requires_customer_id:
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
    def query_agent(self, query: str, custom_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send query to the agent endpoint."""
        
        payload = {
            "input": [{"role": "user", "content": query}]
        }
        
        if custom_inputs:
            payload["custom_inputs"] = custom_inputs
            
        try:
            response = self.deploy_client.predict(
                endpoint=self.endpoint_name,
                inputs=payload
            )
            return response
        except Exception as e:
            print(f"Request failed: {e}")
            raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feedback Generator

# COMMAND ----------

class FeedbackGenerator:
    """Generate synthetic feedback for agent responses."""
    
    def __init__(self):
        self.agent_names = AGENT_NAMES.copy()
        random.shuffle(self.agent_names)  # Randomize order
        
    def get_random_agent(self) -> str:
        """Get a random agent name."""
        return random.choice(self.agent_names)
    
    def generate_feedback(self, query: str, response: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate varied feedback for a query/response pair."""
        feedbacks = []
        agent_name = self.get_random_agent()
        
        # Generate 1-3 different feedback items per query
        num_feedbacks = random.randint(1, 4)
        
        feedback_types = [
            self._generate_helpfulness_feedback,
            self._generate_accuracy_feedback,
            self._generate_clarity_feedback,
            self._generate_completeness_feedback,
            self._generate_response_time_feedback,
        ]
        
        selected_feedback_types = random.sample(feedback_types, min(num_feedbacks, len(feedback_types)))
        
        for feedback_func in selected_feedback_types:
            feedback = feedback_func(query, response, metadata, agent_name)
            if feedback:
                feedbacks.append(feedback)
        
        return feedbacks
    
    def _generate_helpfulness_feedback(self, query: str, response: Dict[str, Any], 
                                     metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate helpfulness feedback (mostly positive)."""
        # 85% positive feedback
        is_helpful = random.random() < 0.85
        
        return {
            "name": "helpfulness",
            "value": is_helpful,
            "source": AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id=agent_name
            ),
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
    
    def _generate_response_time_feedback(self, query: str, response: Dict[str, Any], 
                                       metadata: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Generate response time feedback."""
        # random response time between 1-10 seconds, mostly good
        response_time = random.uniform(1.2, 8.5)
        is_fast_enough = response_time < 6.0
        
        return {
            "name": "response_time_seconds",
            "value": round(response_time, 1),
            "source": AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id=agent_name
            ),
            "rationale": f"Response generated in {response_time:.1f} seconds" + (
                " - within expected range" if is_fast_enough else " - slightly slower than optimal"
            )
        }
    
    def _generate_judge_feedback(self, query: str, response: Dict[str, Any], trace_id: str) -> List[Dict[str, Any]]:
        """Generate judge-based feedback evaluations."""
        feedbacks = []
        
        try:
            # get trace for judge evaluation
            trace = mlflow.get_trace(trace_id)
            if not trace:
                return feedbacks
                
            request = trace.data.spans[0].get_attribute('mlflow.spanInputs')
            response_data = trace.data.spans[0].get_attribute('mlflow.spanOutputs')
            
            # professionalism evaluation
            try:
                professionalism = judges.guideline_adherence(
                    request=request,
                    response=response_data,
                    guidelines=[
                        "The response should be professional and respectful.",
                        "The response should be appropriate for customer service."
                    ],
                    assessment_name="professionalism"
                )
                
                feedbacks.append({
                    "name": "professionalism",
                    "value": professionalism.value == "yes",
                    "source": professionalism.source,
                    "rationale": professionalism.rationale,
                    "trace_id": trace_id
                })
            except Exception as e:
                print(f"Professionalism evaluation failed: {e}")
            
            # accuracy evaluation 
            try:
                accuracy = judges.guideline_adherence(
                    request=request,
                    response=response_data,
                    guidelines=[
                        "The response should provide accurate information based on the customer's query.",
                        "The response should not contain factual errors."
                    ],
                    assessment_name="accuracy"
                )
                
                feedbacks.append({
                    "name": "accuracy", 
                    "value": accuracy.value == "yes",
                    "source": accuracy.source,
                    "rationale": accuracy.rationale,
                    "trace_id": trace_id
                })
            except Exception as e:
                print(f"Accuracy evaluation failed: {e}")
                
        except Exception as e:
            print(f"Judge feedback generation failed: {e}")
            
        return feedbacks
    
    def _generate_reading_ease_feedback(self, response: Dict[str, Any], agent_name: str) -> List[Dict[str, Any]]:
        """Generate reading ease feedback using textstat."""
        feedbacks = []
        
        try:
            # extract response text
            response_text = self._extract_response_text(response)
            if not response_text:
                return feedbacks
                
            # calculate reading ease
            reading_ease = textstat.flesch_reading_ease(response_text)
            
            # categorize reading ease
            if reading_ease >= 90:
                reading_ease_bucket = "Very Easy"
            elif reading_ease >= 80:
                reading_ease_bucket = "Easy"
            elif reading_ease >= 70:
                reading_ease_bucket = "Fairly Easy"
            elif reading_ease >= 60:
                reading_ease_bucket = "Standard"
            elif reading_ease >= 50:
                reading_ease_bucket = "Fairly Difficult"
            elif reading_ease >= 30:
                reading_ease_bucket = "Difficult"
            else:
                reading_ease_bucket = "Very Confusing"
            
            feedbacks.extend([
                {
                    "name": "reading_ease_score",
                    "value": reading_ease,
                    "source": AssessmentSource(
                        source_type=AssessmentSourceType.CODE,
                        source_id="textstat.flesch_reading_ease"
                    ),
                    "rationale": f"Flesch reading ease score: {reading_ease:.1f}"
                },
                {
                    "name": "reading_ease_category",
                    "value": reading_ease_bucket,
                    "source": AssessmentSource(
                        source_type=AssessmentSourceType.CODE,
                        source_id="textstat.flesch_reading_ease"
                    ),
                    "rationale": f"Reading difficulty level: {reading_ease_bucket}"
                }
            ])
            
        except Exception as e:
            print(f"Reading ease calculation failed: {e}")
            
        return feedbacks
    
    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract main response text from agent response."""
        try:
            if "output" in response:
                for item in response["output"]:
                    if item.get("type") == "message" and "content" in item:
                        for content in item["content"]:
                            if content.get("type") == "output_text":
                                return content.get("text", "")
            return ""
        except Exception:
            return ""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic Query Engine

# COMMAND ----------

class SyntheticQueryEngine:
    """Engine for generating and executing synthetic queries."""
    
    def __init__(self, num_queries: int = QUERIES_PER_BATCH):
        self.num_queries = num_queries
        self.generator = QueryGenerator()
        self.client = TelcoAgentClient()
        self.feedback_generator = FeedbackGenerator()
        
    def generate_query_batch(self) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Generate a batch of diverse queries."""
        queries = []
        
        # distribute queries across categories
        categories = list(QUERY_TEMPLATES.keys())
        queries_per_category = self.num_queries // len(categories)
        extra_queries = self.num_queries % len(categories)
        
        for i, category in enumerate(categories):
            # calculate number of queries for category
            category_query_count = queries_per_category
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
        
        random.shuffle(queries)
        return queries
    
    def execute_query_batch(self, queries: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
                           max_workers: int = MAX_WORKERS) -> List[Dict[str, Any]]:
        """Execute batch of queries with parallel processing."""
        
        results = []
        
        def execute_single_query(query_data):
            query, custom_inputs, metadata = query_data
            start_time = time.time()
            
            try:
                print(f"Executing: {query[:100]}...")
                response = self.client.query_agent(query, custom_inputs)
                
                execution_time = time.time() - start_time
                
                # generate and log feedback
                feedbacks = self.feedback_generator.generate_feedback(query, response, metadata)
                
                try:
                    for feedback in feedbacks:
                        mlflow.log_feedback(
                            name=feedback["name"],
                            value=feedback["value"],
                            source=feedback["source"],
                            rationale=feedback["rationale"]
                        )
                except Exception as e:
                    print(f"Failed to log feedback: {e}")
                
                return {
                    "query": query,
                    "custom_inputs": custom_inputs,
                    "metadata": metadata,
                    "response": response,
                    "execution_time": execution_time,
                    "feedbacks": feedbacks,
                    "success": True,
                    "error": None
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                print(f"Query failed: {error_msg}")
                
                return {
                    "query": query,
                    "custom_inputs": custom_inputs,
                    "metadata": metadata,
                    "response": None,
                    "execution_time": execution_time,
                    "feedbacks": [],
                    "success": False,
                    "error": error_msg
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
        
        if successful_queries > 0:
            avg_execution_time = sum(r["execution_time"] for r in results if r["success"]) / successful_queries
        else:
            avg_execution_time = 0
        
        # category breakdown
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
        
        print(f"\nCategory Breakdown:")
        for category, stats in category_stats.items():
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"  {category.title()}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        total_feedbacks = sum(len(r["feedbacks"]) for r in results if r["success"])
        print(f"\nFeedback Generated: {total_feedbacks} items")
        
        if failed_queries > 0:
            print(f"\nFailed Queries:")
            for result in results:
                if not result["success"]:
                    print(f"  - {result['query'][:80]}... | Error: {result['error']}")