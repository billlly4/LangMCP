from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a:int, b:int) -> int:
    """This function adds to numbers"""
    return a+b

@mcp.tool()
def multiple(a:int, b:int) -> int:
    """this function multiplys two numbers"""
    return a*b

if __name__ == "__main__":
    mcp.run(transport="stdio")