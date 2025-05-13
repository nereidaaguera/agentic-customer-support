"""Knowledge base data generator with LLM integration.

This module contains the KnowledgeGenerator class for generating knowledge base
articles and support tickets using foundation models.
"""

from datetime import date, datetime, timedelta
from typing import Any

from databricks.sdk import WorkspaceClient
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import (
    DateType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from telco_support_agent.data.generators.base import BaseGenerator


class KnowledgeGenerator(BaseGenerator):
    """Generator for knowledge base data using foundation models.

    Generates data for knowledge base articles and support tickets.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the knowledge base generator.

        Args:
            config: Config dictionary.
        """
        super().__init__(config)

        self.w = WorkspaceClient()
        self.openai_client = self.w.serving_endpoints.get_open_ai_client()

        # prompt templates for different content types
        self._initialize_prompt_templates()

    def _initialize_prompt_templates(self) -> None:
        """Initialize prompt templates for knowledge base article generation."""
        self.kb_prompts = {
            "FAQ": {
                "Billing": "Create a detailed FAQ entry for a telecom customer support knowledge base about billing. The topic is: {topic}. Format with a clear question as title and a comprehensive answer that would help a customer understand their bill.",
                "Technical": "Create a detailed FAQ entry for a telecom customer support knowledge base about technical issues. The topic is: {topic}. Format with a clear question as title and step-by-step troubleshooting instructions.",
                "Account": "Create a detailed FAQ entry for a telecom customer support knowledge base about account management. The topic is: {topic}. Format with a clear question as title and a detailed explanation of the account procedures.",
                "Services": "Create a detailed FAQ entry for a telecom customer support knowledge base about service features. The topic is: {topic}. Format with a clear question as title and detailed information about how the service works.",
            },
            "Policy": {
                "Billing": "Create a telecom company policy document about billing. The topic is: {topic}. Format with a clear title and structured sections explaining the policy details.",
                "Technical": "Create a telecom company policy document about technical services. The topic is: {topic}. Format with a clear title and structured sections explaining the policy details.",
                "Account": "Create a telecom company policy document about account management. The topic is: {topic}. Format with a clear title and structured sections explaining the policy details.",
                "Services": "Create a telecom company policy document about service offerings. The topic is: {topic}. Format with a clear title and structured sections explaining the policy details.",
            },
            "Guide": {
                "Billing": "Create a step-by-step guide for telecom customers about billing. The topic is: {topic}. Format with a clear title and numbered steps that a customer can follow.",
                "Technical": "Create a detailed technical guide for telecom customers. The topic is: {topic}. Format with a clear title and step-by-step instructions for resolving common issues.",
                "Account": "Create a user guide for telecom customers about account management. The topic is: {topic}. Format with a clear title and step-by-step instructions.",
                "Services": "Create a comprehensive guide about telecom services. The topic is: {topic}. Format with a clear title and detailed sections explaining features and usage.",
            },
        }

        # topics by category for article generation
        self.topics = {
            "Billing": [
                "understanding your bill",
                "payment methods",
                "disputing charges",
                "autopay setup",
                "billing cycle changes",
                "payment due dates",
                "late payment fees",
                "refund policy",
            ],
            "Technical": [
                "internet connectivity issues",
                "mobile data troubleshooting",
                "device setup",
                "network coverage",
                "wifi optimization",
                "SIM card issues",
                "device overheating",
                "battery optimization",
            ],
            "Account": [
                "changing account details",
                "adding a line",
                "switching plans",
                "account security",
                "data privacy",
                "account transfer",
                "closing an account",
                "identity verification",
            ],
            "Services": [
                "international roaming",
                "premium content subscriptions",
                "family controls",
                "device protection plans",
                "5G service",
                "data sharing",
                "visual voicemail",
                "call forwarding",
            ],
        }

        # ticket prompt templates
        self.ticket_prompt = (
            "Generate a realistic telecom customer support ticket description for the category: {category}. "
            + "The ticket should be about: {topic}. Make it sound like something a customer would actually say "
            + "when contacting support, including specific details and context."
        )

        self.resolution_prompt = (
            "Generate a realistic resolution response from a customer service agent for this support ticket: "
            + "'{description}'. The ticket category is {category}. Make it sound professional yet personable, "
            + "and include specific actions taken to resolve the issue."
        )

    def _generate_content_with_llm(self, prompt: str) -> Any:
        """Generate content using foundation model.

        Args:
            prompt: The prompt for content generation.

        Returns:
            Generated content as string.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="databricks-claude-3-7-sonnet",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a telecom industry expert creating content for a customer support knowledge base.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            # extract content from response
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content with LLM: {e}")
            return f"Information about {prompt.lower()}"

    def generate_kb_articles(self) -> DataFrame:
        """Generate knowledge base articles using foundation models.

        Returns:
            DataFrame containing generated knowledge base articles.
        """
        kb_content_types = self.config["distributions"]["kb_content_types"]
        kb_categories = self.config["distributions"]["kb_categories"]

        count = self.config["volumes"]["kb_articles"]

        kb_ids = self.generate_id("KB", 1001, count)

        data = []

        for kb_id in kb_ids:
            # Select content type based on distribution
            content_type = self.select_weighted(kb_content_types)

            # Select category based on distribution
            category = self.select_weighted(kb_categories)

            # Select a random topic for this category
            if category in self.topics:
                topic = self.random.choice(self.topics[category])
            else:
                topic = "general information"

            if (
                content_type in self.kb_prompts
                and category in self.kb_prompts[content_type]
            ):
                prompt = self.kb_prompts[content_type][category].format(topic=topic)

                generated_content = self._generate_content_with_llm(prompt)

                # extract title and content
                lines = generated_content.strip().split("\n")
                title = lines[0].strip("# ")
                if (
                    len(title) > 100
                ):  # if first line is too long, create a summarized title
                    title = f"{topic.title()} Information"

                content = generated_content
                subcategory = topic.capitalize()
            else:
                # fallback if no prompt template exists
                title = f"{category} {content_type}"
                content = (
                    f"Information about {category.lower()} for customer reference."
                )
                subcategory = "General"

            # generate tags
            tags_list = [category.lower(), subcategory.lower(), content_type.lower()]
            # add few random relevant tags
            potential_tags = [
                "mobile",
                "wireless",
                "customer",
                "support",
                "telecom",
                "phone",
                "device",
                "plan",
                "service",
                "account",
                "payment",
            ]
            tags_list.extend(self.random.sample(potential_tags, 3))
            tags = ",".join(tags_list)

            # Generate last updated date (within last year)
            days_ago = self.random.randint(1, 365)
            last_updated = date.today() - timedelta(days=days_ago)

            data.append(
                (
                    kb_id,
                    content_type,
                    category,
                    subcategory,
                    title,
                    content,
                    tags,
                    last_updated,
                )
            )

        schema = StructType(
            [
                StructField("kb_id", StringType(), False),
                StructField("content_type", StringType(), False),
                StructField("category", StringType(), False),
                StructField("subcategory", StringType(), False),
                StructField("title", StringType(), False),
                StructField("content", StringType(), False),
                StructField("tags", StringType(), False),
                StructField("last_updated", DateType(), False),
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df

    def generate_tickets(
        self, customers_df: DataFrame, subscriptions_df: DataFrame
    ) -> DataFrame:
        """Generate support ticket data using foundation models.

        Args:
            customers_df: DataFrame containing customer data.
            subscriptions_df: DataFrame containing subscription data.

        Returns:
            DataFrame containing generated support ticket data.
        """
        # Get config distributions
        ticket_categories = self.config["distributions"]["ticket_categories"]
        ticket_priorities = self.config["distributions"]["ticket_priorities"]
        ticket_statuses = self.config["distributions"]["ticket_statuses"]

        # Get ticket count from config
        count = self.config["volumes"]["tickets"]

        # Generate ticket IDs
        ticket_ids = self.generate_id("TICK", 8001, count)

        # Collect active customers and subscriptions
        customers = customers_df.filter("customer_status = 'Active'").collect()
        subscriptions = subscriptions_df.filter("status = 'Active'").collect()

        # Create customer-subscription mapping for easy lookup
        customer_subscriptions: dict[str, list[Row]] = {}
        for subscription in subscriptions:
            customer_id = subscription.customer_id
            if customer_id not in customer_subscriptions:
                customer_subscriptions[customer_id] = []
            customer_subscriptions[customer_id].append(subscription)

        # Filter customers with active subscriptions
        active_customers = [
            c for c in customers if c.customer_id in customer_subscriptions
        ]

        data = []

        for ticket_id in ticket_ids:
            # Randomly select a customer with active subscriptions
            if not active_customers:
                continue

            customer = self.random.choice(active_customers)
            customer_id = customer.customer_id

            # Select a subscription for this customer
            customer_subs = customer_subscriptions.get(customer_id, [])
            if not customer_subs:
                continue

            subscription = self.random.choice(customer_subs)
            subscription_id = subscription.subscription_id

            # Select category based on distribution
            category = self.select_weighted(ticket_categories)

            # Select priority based on distribution
            priority = self.select_weighted(ticket_priorities)

            # Select status based on distribution
            status = self.select_weighted(ticket_statuses)

            # Generate created date (within last 90 days)
            days_ago = self.random.randint(0, 90)
            created_date = datetime.now() - timedelta(days=days_ago)

            # Select topic for this ticket category
            if category in self.topics:
                topic = self.random.choice(self.topics[category])
            else:
                topic = "general assistance"

            # Generate ticket description using LLM
            prompt = self.ticket_prompt.format(category=category, topic=topic)
            description = self._generate_content_with_llm(prompt)

            # Generate resolved date and resolution for resolved or closed tickets
            resolved_date = None
            resolution = None
            agent_id = None

            if status in ["Resolved", "Closed"]:
                # Resolved date between created date and now
                days_to_resolution = self.random.randint(0, min(7, days_ago))
                resolved_date = created_date + timedelta(days=days_to_resolution)

                # Generate resolution using LLM
                resolution_prompt = self.resolution_prompt.format(
                    description=description, category=category
                )
                resolution = self._generate_content_with_llm(resolution_prompt)

                # Generate agent ID
                agent_id = f"AGT-{self.random.randint(1000, 9999)}"

            data.append(
                (
                    ticket_id,
                    customer_id,
                    subscription_id,
                    created_date,
                    status,
                    category,
                    priority,
                    description,
                    resolution,
                    resolved_date,
                    agent_id,
                )
            )

        schema = StructType(
            [
                StructField("ticket_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("subscription_id", StringType(), False),
                StructField("created_date", TimestampType(), False),
                StructField("status", StringType(), False),
                StructField("category", StringType(), False),
                StructField("priority", StringType(), False),
                StructField("description", StringType(), False),
                StructField("resolution", StringType(), True),  # Nullable
                StructField("resolved_date", TimestampType(), True),  # Nullable
                StructField("agent_id", StringType(), True),  # Nullable
            ]
        )

        df = self.create_dataframe_from_schema(schema, data)

        return df
