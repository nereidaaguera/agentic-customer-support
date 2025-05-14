"""Knowledge base data generator with LLM integration.

This module contains the KnowledgeGenerator class for generating knowledge base
articles and support tickets using foundation models.
"""
from datetime import date, datetime, timedelta
from typing import Any, Optional

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
    Uses LLM to create varied content types and support tickets that reflect common customer scenarios.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the knowledge base generator.

        Args:
            config: Config dictionary containing volumes, distributions, and other settings.
        """
        super().__init__(config)

        self.w = WorkspaceClient()
        self.openai_client = self.w.serving_endpoints.get_open_ai_client()

        self._initialize_prompt_templates()

    def _initialize_prompt_templates(self) -> None:
        """Initialize prompt templates for knowledge base article and ticket generation.

        Sets up detailed prompts for different content types and categories,
        and defines comprehensive topic lists for each category.
        """
        self.kb_prompts = {
            "FAQ": {
                "Billing": (
                    "Create a detailed FAQ entry for a telecom customer support knowledge base about billing. "
                    "The topic is: {topic}. Format with a clear question as title and a comprehensive answer that would help a customer "
                    "understand their bill. Include specific examples of charges, calculations, or plan tiers where relevant. "
                    "Use a conversational, helpful tone and anticipate follow-up questions the customer might have."
                ),
                "Technical": (
                    "Create a detailed FAQ entry for a telecom customer support knowledge base about technical issues. "
                    "The topic is: {topic}. Format with a clear question as title and step-by-step troubleshooting instructions. "
                    "Include common error messages, device-specific instructions (for both iOS and Android where applicable), "
                    "and clear indicators of when the customer should contact support for additional help."
                ),
                "Account": (
                    "Create a detailed FAQ entry for a telecom customer support knowledge base about account management. "
                    "The topic is: {topic}. Format with a clear question as title and a detailed explanation of the account procedures. "
                    "Include references to specific verification requirements, timeframes for changes to take effect, "
                    "and any potential impacts on billing or services that customers should be aware of."
                ),
                "Services": (
                    "Create a detailed FAQ entry for a telecom customer support knowledge base about service features. "
                    "The topic is: {topic}. Format with a clear question as title and detailed information about how the service works. "
                    "Include compatibility information with different plans and devices, activation requirements, "
                    "and practical examples of how customers can maximize value from the service."
                ),
            },
            "Policy": {
                "Billing": (
                    "Create a telecom company policy document about billing. The topic is: {topic}. "
                    "Format with a clear title and structured sections explaining the policy details. "
                    "Include specific timeframes, obligations, customer rights, and company procedures. "
                    "The language should be precise but accessible, with clear definitions of technical terms."
                ),
                "Technical": (
                    "Create a telecom company policy document about technical services. The topic is: {topic}. "
                    "Format with a clear title and structured sections explaining the policy details. "
                    "Include service standards, maintenance windows, customer notification protocols, "
                    "and remediation procedures. Be clear about customer responsibilities versus company obligations."
                ),
                "Account": (
                    "Create a telecom company policy document about account management. The topic is: {topic}. "
                    "Format with a clear title and structured sections explaining the policy details. "
                    "Include identity verification standards, data protection measures, authorized changes, "
                    "and escalation paths. Reference relevant regulatory requirements where applicable."
                ),
                "Services": (
                    "Create a telecom company policy document about service offerings. The topic is: {topic}. "
                    "Format with a clear title and structured sections explaining the policy details. "
                    "Include service level agreements, feature availability guarantees, modification procedures, "
                    "and compatibility requirements. Be specific about how changes are communicated to customers."
                ),
            },
            "Guide": {
                "Billing": (
                    "Create a step-by-step guide for telecom customers about billing. The topic is: {topic}. "
                    "Format with a clear title and numbered steps that a customer can follow. "
                    "Include screenshots descriptions, expected outcomes at each step, and troubleshooting tips "
                    "for common issues. The guide should work for both online account management and mobile app users."
                ),
                "Technical": (
                    "Create a detailed technical guide for telecom customers. The topic is: {topic}. "
                    "Format with a clear title and step-by-step instructions for resolving common issues. "
                    "Include device-specific variations, prerequisite checks, expected behavior after each step, "
                    "and clear indicators of success. Use plain language while accurately describing technical concepts."
                ),
                "Account": (
                    "Create a user guide for telecom customers about account management. The topic is: {topic}. "
                    "Format with a clear title and step-by-step instructions. Include security considerations, "
                    "verification requirements, necessary documentation, and expected processing times. "
                    "Provide alternate methods (online, phone, in-store) where applicable."
                ),
                "Services": (
                    "Create a comprehensive guide about telecom services. The topic is: {topic}. "
                    "Format with a clear title and detailed sections explaining features and usage. "
                    "Include setup instructions, compatibility information, optimization tips, "
                    "and examples of how to leverage advanced features. Address both basic and power user needs."
                ),
            },
            "Procedure": {
                "Billing": (
                    "Create a detailed procedure for handling {topic} in a telecom customer support context. "
                    "Format with a clear title and numbered steps for agents to follow when assisting customers. "
                    "Include verification requirements, system interactions, approval thresholds, documentation needs, "
                    "and escalation paths. The procedure should balance customer satisfaction with company policy compliance."
                ),
                "Technical": (
                    "Create a step-by-step technical procedure for resolving {topic}. Format as an internal support document "
                    "with clear diagnostic steps, required tools or system access, expected outcomes, and troubleshooting "
                    "decision trees. Include escalation criteria and references to relevant technical documentation. "
                    "The procedure should help agents efficiently resolve customer technical issues."
                ),
                "Account": (
                    "Create a detailed procedure for managing {topic} in customer accounts. Format as an internal support "
                    "document with structured steps, verification protocols, system updates required, compliance checks, "
                    "and customer notification requirements. Include examples of proper documentation and quality assurance steps. "
                    "The procedure should ensure consistent, compliant account management."
                ),
                "Services": (
                    "Create a comprehensive procedure for managing {topic} related to telecom services. Format as an internal "
                    "support document with provisioning steps, compatibility verification, testing protocols, and customer "
                    "setup assistance guidelines. Include troubleshooting for common implementation issues and service "
                    "optimization recommendations. The procedure should ensure successful service delivery."
                ),
            },
        }

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
                "unexpected charges explanation",
                "prorated billing",
                "credit application",
                "bill payment options",
                "service interruption credits",
                "third-party charges",
                "family plan billing",
                "taxes and regulatory fees",
                "paper bill vs. electronic billing",
                "payment arrangements",
                "billing error resolution",
                "prepaid vs. postpaid billing differences",
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
                "5G connection troubleshooting",
                "eSIM activation issues",
                "signal strength problems",
                "data throttling",
                "network outages",
                "device software updates",
                "phone not receiving calls",
                "device blacklist removal",
                "mobile hotspot setup",
                "bluetooth connectivity problems",
                "VoLTE configuration",
                "Wi-Fi calling setup",
                "cellular data not working",
                "international roaming setup",
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
                "port-in process",
                "port-out requirements",
                "primary account holder changes",
                "authorized user management",
                "account suspension process",
                "reactivating service",
                "account PINs and passwords",
                "billing responsibility transfer",
                "account statements access",
                "multi-line account management",
                "account merger process",
                "digital account tools",
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
                "mobile hotspot usage",
                "unlimited plan throttling policy",
                "stream quality settings",
                "Wi-Fi calling features",
                "number blocking",
                "spam call filtering",
                "premium messaging services",
                "roaming partners worldwide",
                "prepaid service renewal",
                "data rollover features",
                "reward programs",
                "premium support options",
                "device upgrade eligibility",
            ],
        }

        self.ticket_prompt = (
            "Generate a realistic telecom customer support ticket description for the category: {category}. "
            "The ticket should be about: {topic} and has been assigned the ticket number {ticket_id}. "
            "Begin the description with 'Re: Ticket {ticket_id}' and then continue with the customer's message. "
            "Make it reference specific device models (like iPhone 15, Galaxy S24), "
            "plan names, or specific error messages a customer might see. Include realistic customer sentiment "
            "(frustrated, confused, angry, etc.) and convey the urgency level. Make it sound like something a "
            "customer would actually say when contacting support, including specific details and context. "
            "Avoid generic descriptions and include realistic timestamps, locations, or service details "
            "when appropriate for the issue. DO NOT create any additional ticket numbers or reference numbers."
        )

        self.resolution_prompt = (
            "Generate a realistic resolution response from a customer service agent for this support ticket: "
            "'{description}'. The ticket category is {category}. Include specific actions taken: systems accessed, "
            "changes made, credits applied (with exact amounts), troubleshooting steps performed, or recommendations "
            "provided. Reference specific company policies or procedures where relevant. Make it sound professional "
            "yet personable, with appropriate empathy and follow-up actions. Include realistic timestamps and "
            "appropriate escalation notes if relevant."
        )

        # common support scenarios for ticket generation
        self.common_support_scenarios = [
            {
                "category": "Billing",
                "topic": "disputing charges",
                "template": "Customer disputes charge of ${amount} for {service}; provide breakdown and remove unwanted services.",
            },
            {
                "category": "Billing",
                "topic": "service interruption credits",
                "template": "Customer requests bill credit due to {duration} service outage in {location}; confirm eligibility and apply compensation.",
            },
            {
                "category": "Technical",
                "topic": "network coverage",
                "template": "Customer experiencing repeated data outages in {location}; phone shows {signal_bars} bars but no data connectivity.",
            },
            {
                "category": "Technical",
                "topic": "device setup",
                "template": "New {device_model} setup failed; customer needs assistance with {setup_issue}.",
            },
            {
                "category": "Account",
                "topic": "account security",
                "template": "Customer unable to access online account after {attempts} attempts; needs identity verification and credentials reset.",
            },
            {
                "category": "Account",
                "topic": "closing an account",
                "template": "Customer requests to disconnect {service_type} without affecting other services; need to confirm billing changes.",
            },
            {
                "category": "Technical",
                "topic": "eSIM activation issues",
                "template": "Customer's {device_model} eSIM activation failed with error code {error_code}; need assistance completing setup.",
            },
            {
                "category": "Technical",
                "topic": "device blacklist removal",
                "template": "Customer's phone reported as lost/stolen incorrectly; needs review of blacklist status for {device_model} with IMEI {imei_mock}.",
            },
            {
                "category": "Services",
                "topic": "international roaming",
                "template": "Customer traveling to {country} next week needs international plan setup; concerned about data charges.",
            },
            {
                "category": "Technical",
                "topic": "data throttling",
                "template": "Customer complains of slow data speeds after using {data_amount}GB; needs clarification on plan throttling policies.",
            },
        ]

    def _generate_content_with_llm(self, prompt: str) -> Any:
        """Generate content using foundation model.

        Uses Claude Sonnet model to generate high-quality, contextually
        appropriate content based on the provided prompt.

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
                        "content": (
                            "You are a telecom industry expert creating content for a customer support knowledge base. "
                            "Your content should be accurate, helpful, and reflect current telecom industry standards. "
                            "Use clear, concise language while providing comprehensive information. When applicable, "
                            "include references to mobile apps, online account management, and both iOS and Android specifics. "
                            "Balance technical accuracy with customer-friendly language."
                        ),
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

        Creates a diverse set of knowledge base articles across different
        content types and categories.

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
            # Base tags from metadata
            tags_list = [category.lower(), subcategory.lower(), content_type.lower()]

            # Add topic-specific tags
            topic_words = [word for word in topic.split() if len(word) > 3]
            tags_list.extend(topic_words)

            # Add relevant telecom tags
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
                "data",
                "voice",
                "text",
                "subscription",
                "network",
                "coverage",
                "roaming",
                "billing",
                "troubleshooting",
            ]

            # Add tags related to specific scenarios
            if "international" in topic:
                tags_list.append("international")
            if "5g" in topic.lower():
                tags_list.append("5g")
            if "esim" in topic.lower():
                tags_list.append("esim")

            # Add a few random relevant tags
            tags_list.extend(self.random.sample(potential_tags, 3))

            # Remove duplicates and normalize
            tags_list = list({tag.lower() for tag in tags_list})
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

    def _enhance_ticket_with_data_references(
        self,
        description: str,
        customer: Row,
        subscription: Optional[Row] = None,
        plan: Optional[Row] = None,
        device: Optional[Row] = None,
    ) -> str:
        """Enhance ticket description with references to actual customer data.

        Replaces generic terms with specific references to the customer's actual
        subscription details, device, plan, etc. to make tickets more realistic.

        Args:
            description: Original ticket description.
            customer: Customer row with account details.
            subscription: Optional subscription row with plan/device details.
            plan: Optional plan row with plan details.
            device: Optional device row with device details.

        Returns:
            Enhanced description with specific data references.
        """
        # Extract customer segment and loyalty tier
        segment = getattr(customer, "customer_segment", "")
        loyalty_tier = getattr(customer, "loyalty_tier", "")

        # Basic customer info replacement
        description = description.replace("[customer_segment]", segment)
        description = description.replace("[loyalty_tier]", loyalty_tier)

        # Replace generic plan references if plan data is available
        if subscription and plan:
            # Get plan details
            plan_name = getattr(plan, "plan_name", "")
            plan_type = getattr(plan, "plan_type", "")
            data_limit = getattr(plan, "data_limit_gb", 0)

            # Replace generic terms with specific plan details
            description = description.replace("[plan_name]", plan_name)
            description = description.replace("[plan_type]", plan_type)
            description = description.replace("my plan", f"my {plan_name} plan")

            # Add data limit context
            if data_limit == 0:
                description = description.replace("[data_limit]", "unlimited data")
            else:
                description = description.replace(
                    "[data_limit]", f"{data_limit}GB data"
                )

            # Add monthly charge if available
            monthly_charge = getattr(subscription, "monthly_charge", 0)
            if monthly_charge:
                description = description.replace(
                    "[monthly_charge]", f"${monthly_charge:.2f}"
                )

        # Replace device references if device data is available
        if device:
            device_name = getattr(device, "device_name", "")
            manufacturer = getattr(device, "manufacturer", "")

            description = description.replace("[device_name]", device_name)
            description = description.replace("[manufacturer]", manufacturer)
            description = description.replace("my phone", f"my {device_name}")
            description = description.replace("my device", f"my {device_name}")

        return description

    def _generate_common_support_scenario(self) -> dict[str, str]:
        """Generate a specific common support scenario from predefined templates.

        Selects a common support scenario and fills in the details to create a
        realistic support ticket description based on industry-standard issues.

        Returns:
            Dict containing category, topic, and filled-in description.
        """
        # random scenario template
        scenario = self.random.choice(self.common_support_scenarios)

        # Fill in template variables based on scenario
        if scenario["category"] == "Billing":
            filled_template = scenario["template"].format(
                amount=round(self.random.uniform(10, 200), 2),
                service=self.random.choice(
                    [
                        "Premium Content",
                        "International Calls",
                        "Device Protection",
                        "Data Add-On",
                    ]
                ),
                duration=f"{self.random.randint(2, 24)} hour",
                location=self.fake.city(),
            )
        elif scenario["category"] == "Technical":
            filled_template = scenario["template"].format(
                location=self.fake.city(),
                signal_bars=self.random.randint(1, 4),
                device_model=self.random.choice(
                    [
                        "iPhone 15 Pro",
                        "iPhone 16",
                        "Galaxy S24 Ultra",
                        "Galaxy Z Flip",
                        "Pixel 8",
                        "OnePlus 12",
                        "Motorola Edge",
                    ]
                ),
                setup_issue=self.random.choice(
                    [
                        "eSIM activation",
                        "data transfer",
                        "account login",
                        "mobile hotspot setup",
                    ]
                ),
                error_code=f"E{self.random.randint(1000, 9999)}",
                imei_mock=f"{self.random.randint(100000000, 999999999)}",
                data_amount=self.random.randint(10, 100),
            )
        elif scenario["category"] == "Account":
            filled_template = scenario["template"].format(
                attempts=self.random.randint(2, 5),
                service_type=self.random.choice(
                    [
                        "mobile service",
                        "additional line",
                        "hotspot",
                        "device protection plan",
                        "international add-on",
                    ]
                ),
            )
        elif scenario["category"] == "Services":
            filled_template = scenario["template"].format(
                country=self.random.choice(
                    [
                        "Japan",
                        "United Kingdom",
                        "Mexico",
                        "Canada",
                        "Germany",
                        "Italy",
                        "Australia",
                        "Brazil",
                        "South Korea",
                    ]
                )
            )
        else:
            filled_template = scenario["template"]

        return {
            "category": scenario["category"],
            "topic": scenario["topic"],
            "description": filled_template,
        }

    def generate_tickets(
        self,
        customers_df: DataFrame,
        subscriptions_df: DataFrame,
        plans_df: DataFrame = None,
        devices_df: DataFrame = None,
    ) -> DataFrame:
        """Generate support ticket data using LLM.

        Args:
            customers_df: DataFrame containing customer data.
            subscriptions_df: DataFrame containing subscription data.
            plans_df: Optional DataFrame containing plan data for references.
            devices_df: Optional DataFrame containing device data for references.

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

        # Collect plans and devices if provided
        plans = None
        if plans_df is not None:
            plans = plans_df.collect()
            plans_dict = {plan.plan_id: plan for plan in plans}
        else:
            plans_dict = {}

        devices = None
        if devices_df is not None:
            devices = devices_df.collect()
            devices_dict = {device.device_id: device for device in devices}
        else:
            devices_dict = {}

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

        # Decide how many tickets will use common scenarios vs. generated content
        common_scenario_count = min(count // 3, len(self.common_support_scenarios))

        for idx, ticket_id in enumerate(ticket_ids):
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

            # Get plan and device if available
            plan = (
                plans_dict.get(subscription.plan_id)
                if hasattr(subscription, "plan_id")
                else None
            )
            device = (
                devices_dict.get(subscription.device_id)
                if hasattr(subscription, "device_id")
                else None
            )

            # For some tickets, use common scenarios
            if idx < common_scenario_count:
                # Generate from common scenario
                scenario = self._generate_common_support_scenario()
                category = scenario["category"]
                topic = scenario["topic"]
                description = scenario["description"]

                # Enhance with customer data
                description = self._enhance_ticket_with_data_references(
                    description, customer, subscription, plan, device
                )
            else:
                # Select category based on distribution
                category = self.select_weighted(ticket_categories)

                # Select topic for this ticket category
                if category in self.topics:
                    topic = self.random.choice(self.topics[category])
                else:
                    topic = "general assistance"

                # Generate ticket description using LLM
                prompt = self.ticket_prompt.format(
                    ticket_id=ticket_id, category=category, topic=topic
                )
                description = self._generate_content_with_llm(prompt)

                # Enhance with customer data
                description = self._enhance_ticket_with_data_references(
                    description, customer, subscription, plan, device
                )

            # Select priority based on distribution and content
            # Adjust priority based on content indicators
            base_priority = self.select_weighted(ticket_priorities)

            # Escalate priority for certain keywords or conditions
            priority_escalators = [
                "urgent",
                "immediately",
                "emergency",
                "not working",
                "down",
                "outage",
            ]
            if any(
                escalator in description.lower() for escalator in priority_escalators
            ):
                # Escalate priority by one level if possible
                if base_priority == "Low":
                    priority = "Medium"
                elif base_priority == "Medium":
                    priority = "High"
                else:
                    priority = "Critical"
            else:
                priority = base_priority

            # Select status based on distribution
            status = self.select_weighted(ticket_statuses)

            # Generate created date (within last 90 days)
            days_ago = self.random.randint(0, 90)
            created_date = datetime.now() - timedelta(days=days_ago)

            # Generate resolved date and resolution for resolved or closed tickets
            resolved_date = None
            resolution = None
            agent_id = None

            if status in ["Resolved", "Closed"]:
                # Resolved date between created date and now
                # Adjust resolution time based on priority
                if priority == "Critical":
                    max_days = min(
                        1, days_ago
                    )  # Critical tickets resolved in 1 day max
                elif priority == "High":
                    max_days = min(3, days_ago)  # High priority within 3 days
                elif priority == "Medium":
                    max_days = min(5, days_ago)  # Medium priority within 5 days
                else:
                    max_days = min(7, days_ago)  # Low priority within 7 days

                days_to_resolution = self.random.randint(0, max_days)
                resolved_date = created_date + timedelta(days=days_to_resolution)

                # Generate resolution using LLM
                resolution_prompt = self.resolution_prompt.format(
                    description=description, category=category
                )
                resolution = self._generate_content_with_llm(resolution_prompt)

                # Generate agent ID with department code prefix
                dept_code = {
                    "Billing": "BIL",
                    "Technical": "TEC",
                    "Account": "ACC",
                    "Services": "SRV",
                }.get(category, "SUP")

                agent_id = f"{dept_code}-{self.random.randint(1000, 9999)}"

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
