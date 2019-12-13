# IGRAPH

tutorial: https://igraph.org/python/doc/tutorial/tutorial.html

https://stackoverflow.com/questions/25303620/does-igraphs-gomory-hu-tree-calculate-the-minimum-cut-tree

最小割：原图的每条边有一个割断它的代价，你需要用最小的代价使得这两个点不连通

## Growing Minimum spanning tree

The generic method manages a set of edges $A$, maintaining the following loop invariant

> Prior to each iteration, $A$ is a subset of some minimum spanning tree.

At each step, determine an edge $(u, v)$ that we can add to $A$ without violating this invariant, in the sense that $A\cup \\{(u, v)\\}$ is also a subset of a minimum spanning tree. Such as edge is called **safe edge** for $A$.

A 
