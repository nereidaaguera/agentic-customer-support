# Databricks notebook source
# MAGIC %md
# MAGIC # Convert Delta Tables to Unity Catalog Volume
# MAGIC
# MAGIC Converts knowledge base and support ticket Delta tables to text files in Unity Catalog volumes.
# MAGIC 
# MAGIC Files will be organized as:
# MAGIC - Knowledge Base: `/knowledge_base/{content_type}/{category}/{kb_id}.txt`
# MAGIC - Support Tickets: `/support_tickets/{year}/{month}/{ticket_id}.txt`

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path
from datetime import datetime
from pyspark.sql.functions import col

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
print(f"Root path: {root_path}")

if root_path:
    sys.path.append(root_path)
    print(f"Added {root_path} to Python path")

# COMMAND ----------

env = "prod"

volume_catalog = f"telco_customer_support_{env}"
volume_schema = "bronze"
volume_name = "tech_support"

volume_path = f"/Volumes/{volume_catalog}/{volume_schema}/{volume_name}"
print(f"Target volume path: {volume_path}")

# Source tables
source_catalog = f"telco_customer_support_{env}"
kb_table = f"{source_catalog}.bronze.knowledge_base"
tickets_table = f"{source_catalog}.bronze.support_tickets"

print(f"Source tables:")
print(f"  Knowledge Base: {kb_table}")
print(f"  Support Tickets: {tickets_table}")

# COMMAND ----------

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure the directory exists in the Unity Catalog volume."""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        raise

def save_kb_article_to_volume(kb_id: str, title: str, content: str, 
                             content_type: str, category: str, 
                             subcategory: str, tags: str, last_updated) -> str:
    """Save knowledge base article as text file to Unity Catalog volume."""
    try:
        # directory structure: /knowledge_base/{content_type}/{category}/
        directory_path = os.path.join(
            volume_path,
            "knowledge_base",
            content_type.lower(),
            category.lower()
        )
        ensure_directory_exists(directory_path)

        # filename: {kb_id}.txt
        filename = f"{kb_id}.txt"
        file_path = os.path.join(directory_path, filename)

        # prep content with metadata header
        file_content = f"""ID: {kb_id}
Type: {content_type}
Category: {category}
Subcategory: {subcategory}
Title: {title}
Tags: {tags}
Last Updated: {last_updated}
Generated: {datetime.now().isoformat()}

---

{content}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        return file_path

    except Exception as e:
        print(f"Error saving KB article {kb_id} to volume: {e}")
        return None

def save_ticket_to_volume(ticket_id: str, customer_id: str, subscription_id: str,
                         category: str, priority: str, status: str,
                         description: str, resolution: str = None,
                         created_date=None, resolved_date=None, 
                         agent_id: str = None) -> str:
    """Save a support ticket as a text file to Unity Catalog volume."""
    try:
        # directory structure: /support_tickets/{year}/{month}/
        if created_date:
            if isinstance(created_date, str):
                created_dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            else:
                created_dt = created_date
            year = created_dt.strftime("%Y")
            month = created_dt.strftime("%m")
        else:
            year = datetime.now().strftime("%Y")
            month = datetime.now().strftime("%m")

        directory_path = os.path.join(
            volume_path,
            "support_tickets",
            year,
            month
        )
        ensure_directory_exists(directory_path)

        # filename: {ticket_id}.txt
        filename = f"{ticket_id}.txt"
        file_path = os.path.join(directory_path, filename)

        # prep content with metadata header
        file_content = f"""Ticket ID: {ticket_id}
Customer ID: {customer_id}
Subscription ID: {subscription_id}
Category: {category}
Priority: {priority}
Status: {status}
Created: {created_date}
"""

        if agent_id:
            file_content += f"Agent ID: {agent_id}\n"
        
        if resolved_date:
            file_content += f"Resolved: {resolved_date}\n"

        file_content += f"\n---\nDESCRIPTION:\n---\n\n{description}\n"

        if resolution:
            file_content += f"\n---\nRESOLUTION:\n---\n\n{resolution}\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        return file_path

    except Exception as e:
        print(f"Error saving ticket {ticket_id} to volume: {e}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Source Tables

# COMMAND ----------

try:
    kb_count = spark.sql(f"SELECT COUNT(*) FROM {kb_table}").collect()[0][0]
    print(f"Knowledge Base articles: {kb_count}")
except Exception as e:
    print(f"Error accessing knowledge base table: {e}")
    kb_count = 0

try:
    tickets_count = spark.sql(f"SELECT COUNT(*) FROM {tickets_table}").collect()[0][0]
    print(f"Support tickets: {tickets_count}")
except Exception as e:
    print(f"Error accessing support tickets table: {e}")
    tickets_count = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Volume Directory Structure

# COMMAND ----------

base_directories = [
    os.path.join(volume_path, "knowledge_base"),
    os.path.join(volume_path, "support_tickets")
]

for directory in base_directories:
    ensure_directory_exists(directory)

print("Base directory structure created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge Base

# COMMAND ----------

if kb_count > 0:
    print(f"Converting {kb_count} knowledge base data...")
    
    kb_df = spark.sql(f"SELECT * FROM {kb_table}")
    
    kb_articles = kb_df.collect()
    
    successful_saves = 0
    failed_saves = 0
    
    for i, article in enumerate(kb_articles):
        try:
            file_path = save_kb_article_to_volume(
                kb_id=article.kb_id,
                title=article.title,
                content=article.content,
                content_type=article.content_type,
                category=article.category,
                subcategory=article.subcategory,
                tags=article.tags,
                last_updated=article.last_updated
            )
            
            if file_path:
                successful_saves += 1
            else:
                failed_saves += 1
                
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{kb_count} articles...")
                
        except Exception as e:
            print(f"Error processing article {article.kb_id}: {e}")
            failed_saves += 1
    
    print(f"Knowledge Base conversion complete:")
    print(f"  Successfully saved: {successful_saves}")
    print(f"  Failed saves: {failed_saves}")
else:
    print("No knowledge base articles to convert")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Support Tickets

# COMMAND ----------

if tickets_count > 0:
    print(f"Converting {tickets_count} support tickets...")
    
    tickets_df = spark.sql(f"SELECT * FROM {tickets_table}")
    
    tickets = tickets_df.collect()
    
    successful_saves = 0
    failed_saves = 0
    
    for i, ticket in enumerate(tickets):
        try:
            file_path = save_ticket_to_volume(
                ticket_id=ticket.ticket_id,
                customer_id=ticket.customer_id,
                subscription_id=ticket.subscription_id,
                category=ticket.category,
                priority=ticket.priority,
                status=ticket.status,
                description=ticket.description,
                resolution=ticket.resolution,
                created_date=ticket.created_date,
                resolved_date=ticket.resolved_date,
                agent_id=ticket.agent_id
            )
            
            if file_path:
                successful_saves += 1
            else:
                failed_saves += 1
                
            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{tickets_count} tickets...")
                
        except Exception as e:
            print(f"Error processing ticket {ticket.ticket_id}: {e}")
            failed_saves += 1
    
    print(f"Support Tickets conversion complete:")
    print(f"  Successfully saved: {successful_saves}")
    print(f"  Failed saves: {failed_saves}")
else:
    print("No support tickets to convert")

# COMMAND ----------

def create_volume_summary():
    """Create a summary of files saved to Unity Catalog volume."""
    try:
        volume_path_obj = Path(volume_path)
        if not volume_path_obj.exists():
            return {"error": f"Volume path does not exist: {volume_path}"}

        summary = {
            "volume_path": volume_path,
            "knowledge_base": {"total": 0, "by_category": {}},
            "support_tickets": {"total": 0, "by_year": {}},
            "total_files": 0,
            "total_size_mb": 0.0,
        }

        kb_path = volume_path_obj / "knowledge_base"
        if kb_path.exists():
            for content_type_dir in kb_path.iterdir():
                if content_type_dir.is_dir():
                    for category_dir in content_type_dir.iterdir():
                        if category_dir.is_dir():
                            category_name = f"{content_type_dir.name}/{category_dir.name}"
                            file_count = len(list(category_dir.glob("*.txt")))
                            summary["knowledge_base"]["by_category"][category_name] = file_count
                            summary["knowledge_base"]["total"] += file_count

        tickets_path = volume_path_obj / "support_tickets"
        if tickets_path.exists():
            for year_dir in tickets_path.iterdir():
                if year_dir.is_dir():
                    year_count = 0
                    for month_dir in year_dir.iterdir():
                        if month_dir.is_dir():
                            month_count = len(list(month_dir.glob("*.txt")))
                            year_count += month_count
                    summary["support_tickets"]["by_year"][year_dir.name] = year_count
                    summary["support_tickets"]["total"] += year_count

        return summary

    except Exception as e:
        return {"error": str(e)}

summary = create_volume_summary()

print("=== VOLUME CONVERSION SUMMARY ===")
print(f"Volume Path: {summary.get('volume_path', 'N/A')}")
print(f"Total Files: {summary.get('total_files', 0)}")
print(f"Total Size: {summary.get('total_size_mb', 0)} MB")
print()

if 'knowledge_base' in summary:
    print(f"Knowledge Base Articles: {summary['knowledge_base']['total']}")
    print("  By Category:")
    for category, count in summary['knowledge_base']['by_category'].items():
        print(f"    {category}: {count} files")
print()

if 'support_tickets' in summary:
    print(f"Support Tickets: {summary['support_tickets']['total']}")
    print("  By Year:")
    for year, count in summary['support_tickets']['by_year'].items():
        print(f"    {year}: {count} files")

if 'error' in summary:
    print(f"Error generating summary: {summary['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample File

# COMMAND ----------

kb_sample_path = Path(volume_path) / "knowledge_base"
if kb_sample_path.exists():
    sample_files = list(kb_sample_path.rglob("*.txt"))[:3]  # first 3 files
    
    print(f"Knowledge Base Sample Files ({len(sample_files)}):")
    for file_path in sample_files:
        print(f"  File: {file_path.name}")
        print(f"  Path: {file_path.relative_to(Path(volume_path))}")
        print(f"  Size: {file_path.stat().st_size} bytes")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("  Preview:")
                for line in lines:
                    print(f"    {line.strip()}")
        except Exception as e:
            print(f"  Error reading file: {e}")
        print()

# support ticket files
tickets_sample_path = Path(volume_path) / "support_tickets"
if tickets_sample_path.exists():
    sample_files = list(tickets_sample_path.rglob("*.txt"))[:3]
    
    print(f"Support Ticket Sample Files ({len(sample_files)}):")
    for file_path in sample_files:
        print(f"  File: {file_path.name}")
        print(f"  Path: {file_path.relative_to(Path(volume_path))}")
        print(f"  Size: {file_path.stat().st_size} bytes")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("  Preview:")
                for line in lines:
                    print(f"    {line.strip()}")
        except Exception as e:
            print(f"  Error reading file: {e}")
        print()
# COMMAND ----------

print("=== CONVERSION COMPLETE ===")
print(f"All data has been successfully converted to Unity Catalog volume:")
print(f"Volume: {volume_path}")
print()
print("You can now use these text files for:")
print("- Vector search indexing")
print("- Training data preparation") 
print("- Content analysis")
print("- External system integration")
print("- Backup and archival")