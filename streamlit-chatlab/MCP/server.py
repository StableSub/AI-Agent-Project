import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from typing import List, Optional 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

mcp = FastMCP("DataAnalysis")

@mcp.prompt()
def default_prompt(message) -> list[base.Message]:
    return [
        base.AssistantMessage(
            "You are a helpful data analysis assistant. \n"
            "Please clearly organize and return the results of the tool calling and the data analysis."
        ),
        base.UserMessage(message),
    ]

if __name__ == "__main__":
    mcp.run(transport="stdio")