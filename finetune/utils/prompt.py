

FIXED_PROMPT = """
How do you solve a problem with two different body sources in the same element?
"""

FIXED_PROMPT_BASE_MODEL_COMPLETION = """
Solving a problem with multiple body sources in the same element is a common scenario in the Finite Element Method (FEM). 
In FEM, we typically assume that the element is homogeneous, meaning that the material properties are constant within the element. 
However, when dealing with multiple body sources, such as heat sources or electric charges, the problem becomes more complex.\n
To solve this problem, we can use a technique called ""superposition"" or ""superposition principle."" 
This principle states that the total solution can be obtained by summing the solutions due to each individual source.\n
Mathematically, let's consider a simple example of a 2D heat conduction problem with two heat sources in the same element. 
We want to find the temperature distribution, $T(x,y)$, in a 2D domain $\Omega$ with two heat sources, $Q_1$ and $Q_2$, located at points $(x_1, y_1)$ and $(x_2, y_2)$, respectively.\n
We can write the heat equation as:\n
$$\nabla \cdot (k \nabla T) + Q = 0$$\n
where $k$ is the thermal conductivity, and $Q$ is the heat source term.\n
To apply the superposition principle, we can write the total heat source term as the sum of two individual source terms:\n
$$Q = Q_1 \delta(x-x_1, y-y_1) + Q_2 \delta(x-x_2, y-y_2)$$\n
where $\delta(x-x_1, y-y_1)$ and $\delta(x-x_2, y-y_2)$ are Dirac delta functions, which represent the location of the heat sources.\n
The solution to the heat equation can be written as:\n
$$T(x,y) = T_1(x,y) + T_2(x,y)$$\n
where $T_1(x,y)$ and $T_2(x,y)$ are the solutions due to each individual heat source.\n
We can solve each individual problem separately using the FEM, and then sum the solutions to obtain the total solution.\n
For example, let's consider a simple 2D element with two heat sources, $Q_1 = 100$ W/m$^3$ and $Q_2 = 50$ W/m$^3$, located at $(0.5, 0.5)$ and $(0.8, 0.8)$, respectively. The thermal
"""

FEM_SYSTEM_PROMPT = """
You are an AI professor for a Finite Element Method (FEM) course.
You are asked a question by a student and return an appropriate answer based on course material.
Your response focuses on FEM fundamentals, theories, and applications as presented in the course.
Use standard latex notation when replying with mathematical notation.
"""

DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant helping a user with a question.
"""