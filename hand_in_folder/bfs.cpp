#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define IN_FRONTIER 1
#define NOT_IN_FRONTIER 0
#define HYBRID_THRESHOLD 0.3

void show_graph_edge(Graph g){
    int numNodes = num_nodes(g);

    printf("----------------------incoming:\n");
    for(int i=0; i<numNodes; i++){
        int node = i;
        const Vertex *start = incoming_begin(g, i);
        const Vertex *end = incoming_end(g, i);

        printf("----------------------\n");
        printf("我是%d-node，以下是我的incoming 節點\n", node);
        for (const Vertex *v = start; v!=end; v++) { //iterate all incoming edge
          printf("%d  ", *v);
        }

        printf("\n\n");
    }

    printf("----------------------ougoing:\n");
    for(int i=0; i<numNodes; i++){
        int node = i;
        const Vertex *start = outgoing_begin(g, i);
        const Vertex *end = outgoing_end(g, i);

        printf("----------------------\n");
        printf("我是%d-node，以下是我的ougoing 節點\n", node);
        for (const Vertex *v = start; v!=end; v++) { //iterate all incoming edge
          printf("%d  ", *v);
        }

        printf("\n\n");
    }

    // printf("----------------------incoming:\n");
}

void vertex_set_clear(vertex_set *list) //將set中的vertices設定為0
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}



// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier. 將所有的frontier中的節點的unvisited鄰居丟進 new_frontier中

void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances
)
{
    int num_of_thread = omp_get_max_threads();
    int newfrontier_count[num_of_thread][16] = {0}; // assuming cache line=64bytes，所以padding

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // printf("I'm the %d-th thread\n", thread_id);


        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < frontier->count; i++)
        {
            
            int index = 0;
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    distances[outgoing] = distances[node] + 1;

                    //確保index值有同步
                    while(! __sync_bool_compare_and_swap(&(new_frontier->count), index, index+1) ){
                        index = new_frontier->count;
                    }
                    
                    new_frontier->vertices[index] = outgoing;
                }
            }
        }
    }
    
}


void top_down_step_for_hybrid(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *in_frontier,
    int *new_in_frontier
)
{
    int num_of_thread = omp_get_max_threads();
    int newfrontier_count[num_of_thread][16] = {0}; // assuming cache line=64bytes，所以padding

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // printf("I'm the %d-th thread\n", thread_id);


        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < frontier->count; i++)
        {
            
            int index = 0;
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    distances[outgoing] = distances[node] + 1;

                    //確保index值有同步
                    while(! __sync_bool_compare_and_swap(&(new_frontier->count), index, index+1) ){
                        index = new_frontier->count;
                    }
                    
                    new_frontier->vertices[index] = outgoing;
                    new_in_frontier[outgoing] = IN_FRONTIER;
                }
            }
        }
    }
    
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol){

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes); //vertices set大小 <= graph nodes數量
    vertex_set_init(&list2, graph->num_nodes); //先將frontier中所有entry皆先設定為0

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int cur_depth=0;
    while (frontier->count != 0)
    {

        #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);
        // top_down_step_depth(graph, frontier, new_frontier, sol->distances, cur_depth);


        #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers 更新frontier
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        cur_depth++;
    }
}

void bottom_up_step_sequential(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *in_frontier,
    int *new_in_frontier
)
{
    int num_of_thread = omp_get_max_threads();
    int numNodes = num_nodes(g);
    // printf("num_nodes = %d\n", numNodes);
    // printf("the root is in the frontier? :%d\n", in_frontier[ROOT_NODE_ID]);

    // show_graph_edge(g);

    for(int i=0; i<numNodes; i++){
        int node = i;
        const Vertex *start = incoming_begin(g, i); //找incoming neighbor
        const Vertex *end = incoming_end(g, i);

        if (distances[i] == NOT_VISITED_MARKER){
            
            for (const Vertex* neighbor = start; neighbor !=end; neighbor++){
                //此neighbor有在這輪frontier中
                if(in_frontier[*neighbor] == IN_FRONTIER){
                    // printf("阿哈，我是%d, my neighbor %d in the frontier\n",node, *neighbor);
                    distances[node] = distances[*neighbor] + 1;
                    
                    //將此node放入new frontier中
                    int index = new_frontier->count++;
                    new_frontier->vertices[index] = node;

                    //設定這兩個節點的 new_in_frontier table
                    new_in_frontier[*neighbor] = NOT_IN_FRONTIER;
                    new_in_frontier[node] = IN_FRONTIER;

                    //已找到parent => can break
                    break;
                }
            }
        }
    }

    in_frontier = new_in_frontier;

    //Todo
    //更新 frontier
    //更新 in_frontier
   
}

void bottom_up_step_parallel(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *in_frontier,
    int *new_in_frontier
)
{
    int num_of_thread = omp_get_max_threads();
    int numNodes = num_nodes(g);
    // printf("num_nodes = %d\n", numNodes);
    // printf("the root is in the frontier? :%d\n", in_frontier[ROOT_NODE_ID]);

    // show_graph_edge(g);
    #pragma omp parallel
    {   
        #pragma omp for schedule(dynamic, 1024)
        for(int i=0; i<numNodes; i++){
            int index=0;
            int node = i;
            const Vertex *start = incoming_begin(g, i); //找incoming neighbor
            const Vertex *end = incoming_end(g, i);

            if (distances[i] == NOT_VISITED_MARKER){
                
                for (const Vertex* neighbor = start; neighbor !=end; neighbor++){
                    //此neighbor有在這輪frontier中
                    if(in_frontier[*neighbor] == IN_FRONTIER){
                        
                        distances[node] = distances[*neighbor] + 1;
                        
                        //將此node放入new frontier中
                        while(! __sync_bool_compare_and_swap(&(new_frontier->count), index, index+1) ){
                            index = new_frontier->count; //確保index值有同步
                        }
                        new_frontier->vertices[index] = node;

                        //設定這兩個節點的 new_in_frontier table
                        new_in_frontier[*neighbor] = NOT_IN_FRONTIER;
                        new_in_frontier[node] = IN_FRONTIER;

                        //已找到parent => can break
                        break;
                    }
                }
            }
        }
    }

    

    in_frontier = new_in_frontier;

    //Todo
    //更新 frontier
    //更新 in_frontier
   
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes); //vertices set大小 <= graph nodes數量
    vertex_set_init(&list2, graph->num_nodes); //先將frontier中所有entry皆先設定為0
    int* in_frontier = (int*)calloc(graph->num_nodes, sizeof(int)); //會全部初始化為0
    int* new_in_frontier = (int*)calloc(graph->num_nodes, sizeof(int)); //會全部初始化為0
    


    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    in_frontier[ROOT_NODE_ID] = IN_FRONTIER; //設定Root node 在frontier中

    int cur_depth=0;
    while (frontier->count != 0)
    {

        #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);

        // printf("-----------------\n");
        // printf("原本的frontier數量:%d\n", frontier->count);
        // bottom_up_step_sequential(graph, frontier, new_frontier, sol->distances, in_frontier, new_in_frontier);
        bottom_up_step_parallel(graph, frontier, new_frontier, sol->distances, in_frontier, new_in_frontier);

        // printf("經過一輪的bottom-up-step後的new frontier數量:%d\n", new_frontier->count);
        // printf("-----------------\n\n");
        // top_down_step_depth(graph, frontier, new_frontier, sol->distances, cur_depth);


        #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers 更新frontier
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        cur_depth++;

        //swap，更新 in_frontier
        int *tmp2 = in_frontier;
        in_frontier = new_in_frontier;
        new_in_frontier = tmp2;
        memset(new_in_frontier, NOT_IN_FRONTIER, graph->num_nodes);
    }
    free(in_frontier);
}


void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes); //vertices set大小 <= graph nodes數量
    vertex_set_init(&list2, graph->num_nodes); //先將frontier中所有entry皆先設定為0
    int* in_frontier = (int*)calloc(graph->num_nodes, sizeof(int)); //會全部初始化為0
    int* new_in_frontier = (int*)calloc(graph->num_nodes, sizeof(int)); //會全部初始化為0


    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    in_frontier[ROOT_NODE_ID] = IN_FRONTIER; //設定Root node 在frontier中

    int cur_depth=0;
    while (frontier->count != 0)
    {

        #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);

        // printf("-----------------\n");
        // printf("原本的frontier數量:%d\n", frontier->count);
        // bottom_up_step_sequential(graph, frontier, new_frontier, sol->distances, in_frontier, new_in_frontier);
        double frontier_ratio = ((double)frontier->count)/((double)graph->num_nodes);
        if(frontier_ratio < HYBRID_THRESHOLD){
            top_down_step_for_hybrid(graph, frontier, new_frontier, sol->distances, in_frontier, new_in_frontier);
        }
        else{
            bottom_up_step_parallel(graph, frontier, new_frontier, sol->distances, in_frontier, new_in_frontier);
        }

        

        // printf("經過一輪的bottom-up-step後的new frontier數量:%d\n", new_frontier->count);
        // printf("-----------------\n\n");
        // top_down_step_depth(graph, frontier, new_frontier, sol->distances, cur_depth);


        #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers 更新frontier
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        cur_depth++;

        //swap，更新 in_frontier
        int *tmp2 = in_frontier;
        in_frontier = new_in_frontier;
        new_in_frontier = tmp2;
        memset(new_in_frontier, NOT_IN_FRONTIER, graph->num_nodes); //需將new_in_frontier全部清為0
    }
    free(in_frontier);
}
