#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ABS(x) ((x)>=0.0)?(x):-(x)




//------------------以下是助教原本給的code------------------------------------
// pageRank --
//
// g:           graph to process (see common/graph.h) 此演算法跑的對象graph
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  int num_threads = omp_get_max_threads();
  int numNodes = num_nodes(g);
  double *out_prob = (double *)malloc(sizeof(double) * numNodes); // 計算出每個節點的 Pr(v)/Outgoing(v)
  double num_no_out_score_sum = 0.0; //此數值會隨著iteration一直被改寫
  double thread_no_out_score_sum[num_threads][8]; // 使用padding 避免false sharing，因為此變數會一直被改寫
  double thread_global_diff[num_threads][8]; // 使用padding 避免false sharing
  double global_diff = 0.0;
  //-------Initialize Score--------------
  double equal_prob = 1.0 / numNodes; //初始機率
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i){
    solution[i] = equal_prob;
  }

  int converged = 0;
  while(converged == 0){
    global_diff = 0.0;
    num_no_out_score_sum = 0.0;

    for (int i = 0;i < num_threads;i++) {
      thread_no_out_score_sum[i][0] = 0.0;
      thread_global_diff[i][0] = 0.0;
    }


    // 先計算好一些數值
    #pragma omp parallel 
    {
      int thread_id = omp_get_thread_num();

      #pragma omp for schedule(dynamic, 1024)
      for (int i = 0;i < numNodes;i++) {
        int num_of_outgoing = outgoing_size(g, i); //取得目前節點i的outgoing edge數量
        
        if (num_of_outgoing == 0) {
          thread_no_out_score_sum[thread_id][0] += solution[i]; //算出所有沒有outgoing edge的節點目前分數總合
        } else {
          out_prob[i] = solution[i] / num_of_outgoing; 
        }
      }
    }
    for (int i = 0; i < num_threads; i++) {
      num_no_out_score_sum += thread_no_out_score_sum[i][0]; //因為前面每個thread只有對自己負責區域中的no outgoing vertex socre加總 => 最後還需要加總所有thread算出的結果
    }

    //----------------開始計算Pr(A) 即為score of each vertex--------------------------------------
    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      
      #pragma omp for schedule(dynamic, 1024) //使用dynamic scheduling 可以加速，避免load unbalance
      for (int i = 0;i < numNodes;i++) {
        const Vertex *start = incoming_begin(g, i);
        const Vertex *end = incoming_end(g, i);
        double sum = 0.0;
        
        
        for (const Vertex *v = start; v!=end; v++) { //iterate all incoming edge
          sum += out_prob[*v];
        }

        //計算出這輪此vertex的score
        sum = (damping*sum) + (1.0-damping)/numNodes + (num_no_out_score_sum*damping/numNodes);
        float diff_abs = solution[i]-sum;
        diff_abs = ABS(diff_abs);
        // if(diff_abs<0) diff_abs = -diff_abs; //因為diff有可能是負數 => 取絕對值
        
        thread_global_diff[thread_id][0] += diff_abs;
        solution[i] = sum;
      }
    }

    //因為每個thread只有算自己負責部分的節點的diff => 所以還須加總所有thread算出來的global_diff_sum
    for (int i = 0;i < num_threads;i++) {
      global_diff += thread_global_diff[i][0];
    }

    if(global_diff >= convergence){
      converged=0;
    }else{
      converged=1; //已收斂
    }

    // printf("global_diff = %f\n", global_diff);

    
  }

  free(out_prob);
  





  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for "all nodes vi" 一個for-loop:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
