from fastmcp import FastMCP

# 初始化mcp
mcp = FastMCP("Math")

# mcp tools
@mcp.tool()
def add(a: float, b:float) -> float:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


@mcp.tool()
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# 启动mcp服务，通信方式设置为stdio
if __name__ == "__main__":
    mcp.run(transport="stdio")