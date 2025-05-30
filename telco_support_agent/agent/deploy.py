from mcp_agent import AGENT, LLM_ENDPOINT_NAME
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution
import mlflow
from mlflow.models.auth_policy import SystemAuthPolicy, UserAuthPolicy, AuthPolicy
from databricks import agents


mlflow.set_registry_uri("databricks-uc")

print("---------Testing the Agent Predict Function---------")
print(AGENT.predict({"messages": [{"role": "user", "content": "Try running the notification stream tool with some default values! Maybe a short interval"}]}))

print()
print("---------Testing the Agent Predict Stream Function---------")
for chunk in AGENT.predict_stream({"messages": [{"role": "user", "content": "Try running the notification stream tool with some default values! Maybe a short interval"}]}):
    print(chunk, "-----------\n")

print()
print("---------Logging the Model to MLflow---------")


resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
system_auth_policy = SystemAuthPolicy(resources=resources)
user_auth_policy = UserAuthPolicy(
    api_scopes=[
        "serving.serving-endpoints"
    ]
)

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="mcp_agent.py",
        pip_requirements=[
            "mlflow",
            "backoff",
            "databricks-openai",
            "git+https://github.com/aravind-segu/databricks-sdk-py.git@mcpSupport#egg=databricks-sdk[mcp]",
            f"databricks-connect=={get_distribution('databricks-connect').version}"
        ],
        auth_policy=AuthPolicy(
            system_auth_policy=system_auth_policy,
            user_auth_policy=user_auth_policy
        ),
        infer_code_paths=True
    )

print("Successfully logged the model to MLflow to run id: ", logged_agent_info.run_id)

print()
print("---------Testing the Logged Model---------")
print(mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Please call the start-notification-stream tool with two notifications and a 1s interval"}]},
    env_manager="uv",
))

print("---------Register the Model to UC Catalog---------")

catalog = "telco_customer_support_dev"
schema = "agent"
model_name = "mcp_support_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

print("---------Successfully registered the model to UC Catalog---------")
print("---------Model Name: ", str(UC_MODEL_NAME) + "---------")
print("---------Model Version: ", str(uc_registered_model_info.version) + "---------")
print("---------Deploying the Agent to a Serving Endpoint---------")

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    environment_vars={
        "MLFLOW_REGISTRY_URI": "databricks-uc"
    }
)

print("---------Finished Deployment---------")
