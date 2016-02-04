/**
 * @file   cvext_maxflow.hpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Nov 20 17:14:39 2012
 * 
 * @brief  
 * 
 * 
 */

////////////////////////////////////////////////////////////////
// Usage:
// This section shows how to use the library to dynamically re-compute
// a minimum cut on the following graphs:

// Orignal Graph 
/*
		        SOURCE
		       /       \
		    78/         \0
		     /      78   \
		   node0 -----> node1
		     |   <-----   |
		     |      0     |
		     \            /
		     0\          /78
		       \        /
		          SINK
*/
// Modified Graphs
/*
//              SOURCE
		       /       \
		    15/         \0
		     /     1     \
		   node0 -----> node1
		     |   <-----   |
		     |     4      |
		     \            /
		     5\          /78
		       \        /
		          SINK
*/
// Modified Graphs

// 		        SOURCE
// 		       /       \
// 		    78/         \0
// 		     /     1     \
// 		   node0 -----> node1
// 		     |   <-----   |
// 		     |     4      |
// 		     \            /
// 		     0\          /78
// 		       \        /
// 		          SINK


///////////////////////////////////////////////////


// /* example.cpp */

// #include <stdio.h>
// #include "graph.h"

// int main()
// {
// 	typedef Graph<int,int,int> GraphType;
// 	GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1); 

// 	g -> add_node(); 
// 	g -> add_node(); 

// 	g -> edit_tweights(0, 78, 0);
// 	g -> edit_tweights(1, 0, 78);
// 	g -> add_edge(0, 1, 78, 0);

// 	int flow = g -> maxflow();

// 	printf("Flow = %d\n", flow);
// 	printf("Minimum cut:\n");
	
// 	if (g->what_segment(0) == GraphType::SOURCE) printf("node0 is in the SOURCE set\n");
// 	else printf("node0 is in the SINK set\n");
	
// 	if (g->what_segment(1) == GraphType::SOURCE) printf("node1 is in the SOURCE set\n");
// 	else printf("node1 is in the SINK set\n");


// 	g -> edit_tweights(0, 15, 5);
// 	/* Re-computing max-flow */
// 	flow = g -> maxflow(true);

// 	/* Printing the results */
// 	printf("Flow = %d\n", flow);
// 	printf("Minimum cut:\n");
	
// 	if (g->what_segment(0) == GraphType::SOURCE) printf("node0 is in the SOURCE set\n");
// 	else printf("node0 is in the SINK set\n");
// 	if (g->what_segment(1) == GraphType::SOURCE) printf("node1 is in the SOURCE set\n");
// 	else printf("node1 is in the SINK set\n");

// 	g -> edit_edge(0,1,1,4);
// 	/* Re-computing max-flow */
// 	flow = g -> maxflow(true);

// 	/* Printing the results */
// 	printf("Flow = %d\n", flow);
// 	printf("Minimum cut:\n");
	
// 	if (g->what_segment(0) == GraphType::SOURCE) printf("node0 is in the SOURCE set\n");
// 	else printf("node0 is in the SINK set\n");
// 	if (g->what_segment(1) == GraphType::SOURCE) printf("node1 is in the SOURCE set\n");
// 	else printf("node1 is in the SINK set\n");


// 	delete g;
// 	return 0;
// }


////////////////////////////////////////////////////////////////
// 	This software library implements the maxflow algorithm
// 	described in

// 		"An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision."
// 		Yuri Boykov and Vladimir Kolmogorov.
// 		In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 
// 		September 2004

// 	This algorithm was developed by Yuri Boykov and Vladimir Kolmogorov
// 	at Siemens Corporate Research. To make it available for public use,
// 	it was later reimplemented by Vladimir Kolmogorov based on open publications.

// 	If you use this software for research purposes, you should cite
// 	the aforementioned paper in any resulting publication.

// 	----------------------------------------------------------------------

// 	REUSING TREES:

// 	Starting with version 3.0, there is a also an option of reusing search
// 	trees from one maxflow computation to the next, as described in

// 		"Efficiently Solving Dynamic Markov Random Fields Using Graph Cuts."
// 		Pushmeet Kohli and Philip H.S. Torr
// 		International Conference on Computer Vision (ICCV), 2005

// 	If you use this option, you should cite
// 	the aforementioned paper in any resulting publication.

#ifndef __CV_EXT_MAXFLOW_H__
#define __CV_EXT_MAXFLOW_H__

#include <string.h>
#include <stdlib.h>
#include <assert.h>

////////////////////////////////////////////////////////////////
/*
	Template classes Block and DBlock
	Implement adding and deleting items of the same type in blocks.

	If there there are many items then using Block or DBlock
	is more efficient than using 'new' and 'delete' both in terms
	of memory and time since
	(1) On some systems there is some minimum amount of memory
	    that 'new' can allocate (e.g., 64), so if items are
	    small that a lot of memory is wasted.
	(2) 'new' and 'delete' are designed for items of varying size.
	    If all items has the same size, then an algorithm for
	    adding and deleting can be made more efficient.
	(3) All Block and DBlock functions are inline, so there are
	    no extra function calls.

	Differences between Block and DBlock:
	(1) DBlock allows both adding and deleting items,
	    whereas Block allows only adding items.
	(2) Block has an additional operation of scanning
	    items added so far (in the order in which they were added).
	(3) Block allows to allocate several consecutive
	    items at a time, whereas DBlock can add only a single item.

	Note that no constructors or destructors are called for items.

	Example usage for items of type 'MyType':

	///////////////////////////////////////////////////
	#include "block.h"
	#define BLOCK_SIZE 1024
	typedef struct { int a, b; } MyType;
	MyType *ptr, *array[10000];

	...

	Block<MyType> *block = new Block<MyType>(BLOCK_SIZE);

	// adding items
	for (int i=0; i<sizeof(array); i++)
	{
		ptr = block -> New();
		ptr -> a = ptr -> b = rand();
	}

	// reading items
	for (ptr=block->ScanFirst(); ptr; ptr=block->ScanNext())
	{
		printf("%d %d\n", ptr->a, ptr->b);
	}

	delete block;

	...

	DBlock<MyType> *dblock = new DBlock<MyType>(BLOCK_SIZE);
	
	// adding items
	for (int i=0; i<sizeof(array); i++)
	{
		array[i] = dblock -> New();
	}

	// deleting items
	for (int i=0; i<sizeof(array); i+=2)
	{
		dblock -> Delete(array[i]);
	}

	// adding items
	for (int i=0; i<sizeof(array); i++)
	{
		array[i] = dblock -> New();
	}

	delete dblock;

	///////////////////////////////////////////////////

	Note that DBlock deletes items by marking them as
	empty (i.e., by adding them to the list of free items),
	so that this memory could be used for subsequently
	added items. Thus, at each moment the memory allocated
	is determined by the maximum number of items allocated
	simultaneously at earlier moments. All memory is
	deallocated only when the destructor is called.
*/

// #ifndef __BLOCK_H__
// #define __BLOCK_H__


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

template <class Type> class Block
{
 public:
  /* Constructor. Arguments are the block size and
     (optionally) the pointer to the function which
     will be called if allocation failed; the message
     passed to this function is "Not enough memory!" */
  Block(int size, void (*err_function)(char *) = NULL) { first = last = NULL; block_size = size; error_function = err_function; }

  /* Destructor. Deallocates all items added so far */
  ~Block() { while (first) { block *next = first -> next; delete first; first = next; } }

  /* Allocates 'num' consecutive items; returns pointer
     to the first item. 'num' cannot be greater than the
     block size since items must fit in one block */
  Type *New(int num = 1)
  {
    Type *t;

    if (!last || last->current + num > last->last)
    {
      if (last && last->next) last = last -> next;
      else
      {
        block *next = (block *) new char [sizeof(block) + (block_size-1)*sizeof(Type)];
        if (!next) { if (error_function) (*error_function)((char*)"Not enough memory!"); exit(1); }
        if (last) last -> next = next;
        else first = next;
        last = next;
        last -> current = & ( last -> data[0] );
        last -> last = last -> current + block_size;
        last -> next = NULL;
      }
    }

    t = last -> current;
    last -> current += num;
    return t;
  }

  /* Returns the first item (or NULL, if no items were added) */
  Type *ScanFirst()
  {
    for (scan_current_block=first; scan_current_block; scan_current_block = scan_current_block->next)
    {
      scan_current_data = & ( scan_current_block -> data[0] );
      if (scan_current_data < scan_current_block -> current) return scan_current_data ++;
    }
    return NULL;
  }

  /* Returns the next item (or NULL, if all items have been read)
     Can be called only if previous ScanFirst() or ScanNext()
     call returned not NULL. */
  Type *ScanNext()
  {
    while (scan_current_data >= scan_current_block -> current)
    {
      scan_current_block = scan_current_block -> next;
      if (!scan_current_block) return NULL;
      scan_current_data = & ( scan_current_block -> data[0] );
    }
    return scan_current_data ++;
  }

  /* Marks all elements as empty */
  void Reset()
  {
    block *b;
    if (!first) return;
    for (b=first; ; b=b->next)
    {
      b -> current = & ( b -> data[0] );
      if (b == last) break;
    }
    last = first;
  }

  /***********************************************************************/

 private:

  typedef struct block_st
  {
    Type					*current, *last;
    struct block_st			*next;
    Type					data[1];
  } block;

  int		block_size;
  block	*first;
  block	*last;

  block	*scan_current_block;
  Type	*scan_current_data;

  void	(*error_function)(char *);
};

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

template <class Type> class DBlock
{
 public:
  /* Constructor. Arguments are the block size and
     (optionally) the pointer to the function which
     will be called if allocation failed; the message
     passed to this function is "Not enough memory!" */
  DBlock(int size, void (*err_function)(char *) = NULL) { first = NULL; first_free = NULL; block_size = size; error_function = err_function; }

  /* Destructor. Deallocates all items added so far */
  ~DBlock() { while (first) { block *next = first -> next; delete first; first = next; } }

  /* Allocates one item */
  Type *New()
  {
    block_item *item;

    if (!first_free)
    {
      block *next = first;
      first = (block *) new char [sizeof(block) + (block_size-1)*sizeof(block_item)];
      if (!first) { if (error_function) (*error_function)((char*)"Not enough memory!"); exit(1); }
      first_free = & (first -> data[0] );
      for (item=first_free; item<first_free+block_size-1; item++)
        item -> next_free = item + 1;
      item -> next_free = NULL;
      first -> next = next;
    }

    item = first_free;
    first_free = item -> next_free;
    return (Type *) item;
  }

  /* Deletes an item allocated previously */
  void Delete(Type *t)
  {
    ((block_item *) t) -> next_free = first_free;
    first_free = (block_item *) t;
  }

  /***********************************************************************/

 private:

  typedef union block_item_st
  {
    Type			t;
    block_item_st	*next_free;
  } block_item;

  typedef struct block_st
  {
    struct block_st			*next;
    block_item				data[1];
  } block;

  int			block_size;
  block		*first;
  block_item	*first_free;

  void	(*error_function)(char *);
};


// NOTE: in UNIX you need to use -DNDEBUG preprocessor option to supress assert's!!!



// captype: type of edge capacities (excluding t-links)
// tcaptype: type of t-links (edges between nodes and terminals)
// flowtype: type of total flow
//
// Current instantiations are in instances.inc
template <typename captype, typename tcaptype, typename flowtype> 
class Graph
{
 public:
  typedef enum
  {
    SOURCE	= 0,
    SINK	= 1
  } termtype; // terminals 
  typedef int node_id;

  /////////////////////////////////////////////////////////////////////////
  //                     BASIC INTERFACE FUNCTIONS                       //
  //              (should be enough for most applications)               //
  /////////////////////////////////////////////////////////////////////////

  // Constructor. 
  // The first argument gives an estimate of the maximum number of nodes that can be added
  // to the graph, and the second argument is an estimate of the maximum number of edges.
  // The last (optional) argument is the pointer to the function which will be called 
  // if an error occurs; an error message is passed to this function. 
  // If this argument is omitted, exit(1) will be called.
  //
  // IMPORTANT: It is possible to add more nodes to the graph than node_num_max 
  // (and node_num_max can be zero). However, if the count is exceeded, then 
  // the internal memory is reallocated (increased by 50%) which is expensive. 
  // Also, temporarily the amount of allocated memory would be more than twice than needed.
  // Similarly for edges.
  // If you wish to avoid this overhead, you can download version 2.2, where nodes and edges are stored in blocks.
  Graph(int node_num_max, int edge_num_max, void (*err_function)(char *) = NULL);

  // Destructor
  ~Graph();

  // Adds node(s) to the graph. By default, one node is added (num=1); then first call returns 0, second call returns 1, and so on. 
  // If num>1, then several nodes are added, and node_id of the first one is returned.
  // IMPORTANT: see note about the constructor 
  node_id add_node(int num = 1);

  // Adds a bidirectional edge between 'i' and 'j' with the weights 'cap' and 'rev_cap'.
  // IMPORTANT: see note about the constructor 
  void add_edge(node_id i, node_id j, captype cap, captype rev_cap);



  // Computes the maxflow. Can be called several times.
  // FOR DESCRIPTION OF reuse_trees, SEE mark_node().
  // FOR DESCRIPTION OF changed_list, SEE remove_from_changed_list().
  flowtype maxflow(bool reuse_trees = false, Block<node_id>** changed_list = NULL);

  // After the maxflow is computed, this function returns to which
  // segment the node 'i' belongs (Graph<captype,tcaptype,flowtype>::SOURCE or Graph<captype,tcaptype,flowtype>::SINK).
  //
  // Occasionally there may be several minimum cuts. If a node can be assigned
  // to both the source and the sink, then default_segm is returned.
  termtype what_segment(node_id i, termtype default_segm = SOURCE);



  //////////////////////////////////////////////
  //       ADVANCED INTERFACE FUNCTIONS       //
  //      (provide access to the graph)       //
  //////////////////////////////////////////////
 private:
  struct node;
  struct arc;
 public:

  ////////////////////////////
  // 1. Reallocating graph. //
  ////////////////////////////

  // Removes all nodes and edges. 
  // After that functions add_node() and add_edge() must be called again. 
  //
  // Advantage compared to deleting Graph and allocating it again:
  // no calls to delete/new (which could be quite slow).
  //
  // If the graph structure stays the same, then an alternative
  // is to go through all nodes/edges and set new residual capacities
  // (see functions below).
  void reset();

  ////////////////////////////////////////////////////////////////////////////////
  // 2. Functions for getting pointers to arcs and for reading graph structure. //
  //    NOTE: adding new arcs may invalidate these pointers (if reallocation    //
  //    happens). So it's best not to add arcs while reading graph structure.   //
  ////////////////////////////////////////////////////////////////////////////////

  // The following two functions return arcs in the same order that they
  // were added to the graph. NOTE: for each call add_edge(i,j,cap,cap_rev)
  // the first arc returned will be i->j, and the second j->i.
  // If there are no more arcs, then the function can still be called, but
  // the returned arc_id is undetermined.
  typedef arc* arc_id;
  arc_id get_first_arc();
  arc_id get_next_arc(arc_id a);

  // other functions for reading graph structure
  int get_node_num() { return node_num; }
  int get_arc_num() { return (int)(arc_last - arcs); }
  void get_arc_ends(arc_id a, node_id& i, node_id& j); // returns i,j to that a = i->j

  ///////////////////////////////////////////////////
  // 3. Functions for reading residual capacities. //
  ///////////////////////////////////////////////////

  // returns residual capacity of SOURCE->i minus residual capacity of i->SINK
  tcaptype get_trcap(node_id i); 
  // returns residual capacity of arc a
  captype get_rcap(arc* a);

  /////////////////////////////////////////////////////////////////
  // 4. Functions for setting residual capacities.               //
  //    NOTE: If these functions are used, the value of the flow //
  //    returned by maxflow() will not be valid!                 //
  /////////////////////////////////////////////////////////////////

  void set_trcap(node_id i, tcaptype trcap); 
  void set_rcap(arc* a, captype rcap);

  // Edit capacity of t-edge when "using" tree-recycling 
  void edit_tweights(node_id i, tcaptype cap_source, tcaptype cap_sink);
	
  // Edit capacity of t-edge when "not using" tree-recycling :		
  // If yoy are editing capacities using this function, "maxflow(false)" needs to be called
  void edit_tweights_wt(node_id i, tcaptype cap_source, tcaptype cap_sink);
	
  // Edit capacity of n-edge when "using" tree-recycling 
  void edit_edge(node_id from, node_id to, captype cap, captype rev_cap);
	
  // Edit capacity of n-edge when "not using" tree-recycling :		
  // If yoy are editing capacities using this function, "maxflow(false)" needs to be called
  void edit_edge_wt(node_id from, node_id to, captype cap, captype rev_cap);
	

  //tcaptype MIN(tcaptype a, tcaptype b);
  //tcaptype MAX(tcaptype a, tcaptype b);
	
  ////////////////////////////////////////////////////////////////////
  // 5. Functions related to reusing trees & list of changed nodes. //
  ////////////////////////////////////////////////////////////////////

  // If flag reuse_trees is true while calling maxflow(), then search trees
  // are reused from previous maxflow computation (unless it's the first call to maxflow()).
  // In this case BEFORE calling maxflow() the user must
  // specify which parts of the graph have changed by calling mark_node():
  //   add_tweights(i),set_trcap(i)    => call mark_node(i)
  //   add_edge(i,j),set_rcap(a)       => call mark_node(i); mark_node(j)
  //
  // This option makes sense only if a small part of the graph is changed.
  // The initialization procedure goes only through marked nodes then.
  // 
  // mark_node(i) can either be called before or after graph modification.
  // Can be called more than once per node, but calls after the first one
  // do not have any effect.
  // 
  // NOTE: 
  //   1. It is not necessary to call mark_node() if the change is ``not essential'',
  //      i.e. sign(trcap) is preserved for a node and zero/nonzero status is preserved for an arc.
  //   2. To check that you marked all necessary nodes, you can call maxflow(true) after calling maxflow(false).
  //      If everything is correct, the two calls must return the same value of flow. (Useful for debugging).
  void mark_node(node_id i);

  // If changed_list is not NULL while calling maxflow(), then the algorithm
  // keeps a list of nodes which could potentially have changed their segmentation label
  // (unless it's the first call to maxflow).
  // In this case AFTER calling maxflow() the user must call remove_from_changed_list()
  // for every node in the list. (Exception: this is necessary only if the next 
  // maxflow computation uses option reuse_trees).
  //
  // Nodes which are not in the list are guaranteed to keep their old segmentation label (SOURCE or SINK).
  //
  // Pointer to the list is returned in changed_list. (See block.h on how to read it).
  //
  // NOTE: The user should not deallocate the returned list. (This will be done by Graph destructor).
  void remove_from_changed_list(node_id i) 
  { 
    assert(i>=0 && i<node_num && nodes[i].is_in_changed_list); 
    nodes[i].is_in_changed_list = 0;
  }






  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
	
 private:
  // internal variables and functions

  struct node
  {
    arc			*first;		// first outcoming arc

    arc			*parent;	// node's parent
    node		*next;		// pointer to the next active node
    //   (or to itself if it is the last node in the list)
    int			TS;			// timestamp showing when DIST was computed
    int			DIST;		// distance to the terminal
    int			is_sink : 1;	// flag showing whether the node is in the source or in the sink tree (if parent!=NULL)
    int			is_marked : 1;	// set by mark_node()
    int			is_in_changed_list : 1; // set by maxflow if 

    tcaptype	tr_cap;		// if tr_cap > 0 then tr_cap is residual capacity of the arc SOURCE->node
    // otherwise         -tr_cap is residual capacity of the arc node->SINK 

    tcaptype	t_cap;
    tcaptype	con_flow;
  };

  struct arc
  {
    node		*head;		// node the arc points to
    arc			*next;		// next arc with the same originating node
    arc			*sister;	// reverse arc

    captype		r_cap;		// residual capacity
    captype		e_cap;		// original capacity
  };

  struct nodeptr
  {
    node    	*ptr;
    nodeptr		*next;
  };
  static const int NODEPTR_BLOCK_SIZE = 128;

  node				*nodes, *node_last, *node_max; // node_last = nodes+node_num, node_max = nodes+node_num_max;
  arc					*arcs, *arc_last, *arc_max; // arc_last = arcs+2*edge_num, arc_max = arcs+2*edge_num_max;

  int					node_num;

  DBlock<nodeptr>		*nodeptr_block;

  void	(*error_function)(char *);	// this function is called if a error occurs,
  // with a corresponding error message
  // (or exit(1) is called if it's NULL)

  flowtype			flow;		// total flow

  // reusing trees & list of changed pixels
  int					maxflow_iteration; // counter
  bool				keep_changed_list;
  Block<node_id>		*changed_list;

  /////////////////////////////////////////////////////////////////////////

  node				*queue_first[2], *queue_last[2];	// list of active nodes
  nodeptr				*orphan_first, *orphan_last;		// list of pointers to orphans
  int					TIME;								// monotonically increasing global counter

  /////////////////////////////////////////////////////////////////////////

  void reallocate_nodes(int num); // num is the number of new nodes
  void reallocate_arcs();

  // functions for processing active list
  void set_active(node *i);
  node *next_active();

  // functions for processing orphans list
  void set_orphan_front(node* i); // add to the beginning of the list
  void set_orphan_rear(node* i);  // add to the end of the list

  void add_to_changed_list(node* i);

  void maxflow_init();             // called if reuse_trees == false
  void maxflow_reuse_trees_init(); // called if reuse_trees == true
  void augment(arc *middle_arc);
  void process_source_orphan(node *i);
  void process_sink_orphan(node *i);

  void test_consistency(node* current_node=NULL); // debug function
};


///////////////////////////////////////
// Implementation - inline functions //
///////////////////////////////////////



template <typename captype, typename tcaptype, typename flowtype> 
inline typename Graph<captype,tcaptype,flowtype>::node_id Graph<captype,tcaptype,flowtype>::add_node(int num)
{
  assert(num > 0);

  if (node_last + num > node_max) reallocate_nodes(num);

  if (num == 1)
  {
    node_last -> first = NULL;
    node_last -> tr_cap = 0;
    node_last -> t_cap = 0;
    node_last -> con_flow = 0;
    node_last -> is_marked = 0;
    node_last -> is_in_changed_list = 0;

    node_last ++;
    return node_num ++;
  }
  else
  {
    memset(node_last, 0, num*sizeof(node));

    node_id i = node_num;
    node_num += num;
    node_last += num;
    return i;
  }
}

template <typename captype, typename tcaptype, typename flowtype> 
inline void Graph<captype,tcaptype,flowtype>::add_edge(node_id _i, node_id _j, captype cap, captype rev_cap)
{
  assert(_i >= 0 && _i < node_num);
  assert(_j >= 0 && _j < node_num);
  assert(_i != _j);
  assert(cap >= 0);
  assert(rev_cap >= 0);

  if (arc_last == arc_max) reallocate_arcs();

  arc *a = arc_last ++;
  arc *a_rev = arc_last ++;

  node* i = nodes + _i;
  node* j = nodes + _j;

  a -> sister = a_rev;
  a_rev -> sister = a;
  a -> next = i -> first;
  i -> first = a;
  a_rev -> next = j -> first;
  j -> first = a_rev;
  a -> head = j;
  a_rev -> head = i;
  a -> r_cap = cap;
  a_rev -> r_cap = rev_cap;
  a -> e_cap = cap;
  a_rev -> e_cap = rev_cap;

}

template <typename captype, typename tcaptype, typename flowtype> 
inline typename Graph<captype,tcaptype,flowtype>::arc* Graph<captype,tcaptype,flowtype>::get_first_arc()
{
  return arcs;
}

template <typename captype, typename tcaptype, typename flowtype> 
inline typename Graph<captype,tcaptype,flowtype>::arc* Graph<captype,tcaptype,flowtype>::get_next_arc(arc* a) 
{
  return a + 1; 
}

template <typename captype, typename tcaptype, typename flowtype> 
inline void Graph<captype,tcaptype,flowtype>::get_arc_ends(arc* a, node_id& i, node_id& j)
{
  assert(a >= arcs && a < arc_last);
  i = (node_id) (a->sister->head - nodes);
  j = (node_id) (a->head - nodes);
}

template <typename captype, typename tcaptype, typename flowtype> 
inline tcaptype Graph<captype,tcaptype,flowtype>::get_trcap(node_id i)
{
  assert(i>=0 && i<node_num);
  return nodes[i].tr_cap;
}

template <typename captype, typename tcaptype, typename flowtype> 
inline captype Graph<captype,tcaptype,flowtype>::get_rcap(arc* a)
{
  assert(a >= arcs && a < arc_last);
  return a->r_cap;
}

template <typename captype, typename tcaptype, typename flowtype> 
inline void Graph<captype,tcaptype,flowtype>::set_trcap(node_id i, tcaptype trcap)
{
  assert(i>=0 && i<node_num); 
  nodes[i].tr_cap = trcap;
}

template <typename captype, typename tcaptype, typename flowtype> 
inline void Graph<captype,tcaptype,flowtype>::set_rcap(arc* a, captype rcap)
{
  assert(a >= arcs && a < arc_last);
  a->r_cap = rcap;
}


template <typename captype, typename tcaptype, typename flowtype> 
inline typename Graph<captype,tcaptype,flowtype>::termtype Graph<captype,tcaptype,flowtype>::what_segment(node_id i, termtype default_segm)
{
  if (nodes[i].parent)
  {
    return (nodes[i].is_sink) ? SINK : SOURCE;
  }
  else
  {
    return default_segm;
  }
}

template <typename captype, typename tcaptype, typename flowtype> 
inline void Graph<captype,tcaptype,flowtype>::mark_node(node_id _i)
{
  node* i = nodes + _i;
  if (!i->next)
  {
    /* it's not in the list yet */
    if (queue_last[1]) queue_last[1] -> next = i;
    else               queue_first[1]        = i;
    queue_last[1] = i;
    i -> next = i;
  }
  i->is_marked = 1;
}

////////////////////////////////////////////////////////////////
//template <typename captype, typename tcaptype, typename flowtype> 
//inline tcaptype Graph<captype,tcaptype,flowtype>::MIN(tcaptype a, tcaptype b)
//{
//  if (a<b) return a;
//  return b;
//}
//
//
//template <typename captype, typename tcaptype, typename flowtype> 
//inline tcaptype Graph<captype,tcaptype,flowtype>::MAX(tcaptype a, tcaptype b)
//{
//  if (a>b) return a;
//  return b;
//}
////////////////////////////////////////////////////////////////

//================================================================
// SEGMENTATION with maxflow 
//================================================================

#include "cvext_c.h"

typedef Graph<int, int, int> CvGraph32s;
typedef Graph<float, float, float> CvGraph32f;

/**
 * 
 *
 ********************************************************************
 * Example:
 ********************************************************************

 int main()
 {
   IplImage * raw = cvLoadImage("test.png",
                                0 //grayscale
                                );
   assert(raw);
   CvMat rawheader;
   CvMat * img = cvGetMat(raw, &rawheader);
   assert(cvGetElemType(img)==CV_8U);

   SHOW(img);
   CvMaxFlowSegmentation<CvGraph32f> engine(img);
   engine->set_boundary_term(xxx);
   engine->set_regional_term(yyy);
   engine->maxflow();
   engine->get_segmentation(segImage);
   SHOW(segImage);

   return 0;
 }

 ********************************************************************
 * 
 */

template <typename GraphType>
class CvMaxFlowSegmentation
{
	CvMat * m_img;
  int ncols;
  int nrows;

  GraphType * m_g;
  float m_sigma2Inv;
  float m_c;

  //V = c*exp(-abs(m(E(:,1))-m(E(:,2))))./(2*sigma^2);
  inline float getEdgeWeight(CvMat * img, int a, int b)
  {
    unsigned char* pImg = (unsigned char*)img->data.ptr;
    float diff = -(double)pow(float(pImg[a]-pImg[b]),2.0f); // magnitude
    return m_c*exp(diff)*m_sigma2Inv*1/* 1-pixel displacement */;    
  }

  inline void addEdgeWeight(CvMat * img, int a, int b)
  {
    float Wa = getEdgeWeight(img, a, b);
    float Wb = getEdgeWeight(img, b, a);
    m_g->add_edge(a, b, Wa, Wb);
  }
  
 public:
  CvMaxFlowSegmentation(const CvArr * src, const int sigma = 2):
	m_g(NULL),
	m_sigma2Inv(1.0/float(2*sigma*sigma)),
	m_c(0xff)
  {
    CvMat imghdr;
    m_img = cvCloneMat(cvGetMat(src, &imghdr));
	ncols = m_img->cols;
	nrows = m_img->rows;

	m_g = new GraphType(/*estimated # of nodes*/ ncols*nrows,
                        /*estimated # of edges*/
                        ncols*(nrows-1)+(ncols-1)*nrows); 
    for (int i = 0; i < ncols*nrows; i++)
      m_g -> add_node(); 
  }
  ~CvMaxFlowSegmentation()
  {
    delete m_g;
	cvReleaseMatEx(m_img);
  }

  /** 
   * regional term that defines source/target distance for each pixel
   *
   * typically,
   * computed by negtive log-likelihood of conditional probability
   *         R_p('obj')=-log(Pr(Ip|"obj")); 
   *         R_p('bg') =-log(Pr(Ip|"bg"));
   * 
   * @param toSrc IN: CV_64F, size of HxW
   * @param toTar IN: CV_64F, size of HxW
   *
   *                64-bit floating-point data 
   *                first channel term for distance to source
   *                second channel term for distance to target
   *
   * @param use_ne_loglik in: whether compute negtive likelihood 
   */
  CVStatus set_regional_term(const CvArr * toSrcArr,
                             const CvArr * toTarArr,
                             int use_ne_loglik = 1)
  {
    CV_FUNCNAME("CvMaxFlowSegmentation::set_regional_term");
    CvMat srcArrHdr, tarArrHdr;
    CvMat * toSrc = cvGetMat(toSrcArr, &srcArrHdr);
    CvMat * toTar = cvGetMat(toTarArr, &tarArrHdr);
    __BEGIN__;
    int nrows = toSrc->rows;
    int ncols = toSrc->cols;
	for (int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols; j++){
		double dist_s; double dist_t; 
		if (CV_MAT_TYPE(toSrc->type)==CV_64F ||
            CV_MAT_TYPE(toSrc->type)==CV_32F)
        {
			dist_s = cvmGet(toSrc, i, j); // distance to source
			dist_t = cvmGet(toTar, i, j); // distance to target
		}else if (CV_MAT_TYPE(toSrc->type)==CV_8U){
			dist_s = CV_MAT_ELEM(*toSrc, uchar, i, j);
			dist_t = CV_MAT_ELEM(*toTar, uchar, i, j);
		}else {assert(false);}

        if (use_ne_loglik)
        {
          dist_s = -log(dist_s);
          dist_t = -log(dist_t);
        }
        m_g->edit_tweights(i*ncols+j, dist_s, dist_t);
      }
    }
    __END__;
	return CV_StsOk;
  }

  /** 
   * boundary term that defines boundary term between pixels
   * both horizontal and vertical distance
   *
   * typically,
   * computed by
   *   B_pq = c*exp(-abs(m(E(:,1)),m(E(:,2))))./(2*sigma^2);
   * 
   * @param arr IN: CV_64F, size of (H-1)x(W-1)
   *
   *                weight of boundary, typically between
   *                each neighboring pixels
   *                (8-neighbor in 2D case)
   */
  CVStatus set_boundary_term(const CvArr * arr)
  {
    CV_FUNCNAME("CvMaxFlowSegmentation::set_boundary_term");
    CvMat arrhdr;
    __BEGIN__;
    CvMat * boundary_term = cvGetMat(arr, &arrhdr);
    
    // horizontal
    for (int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols-1; j++){
        addEdgeWeight(boundary_term, i*ncols+j, i*ncols+(j+1));
      }    
    }

    // vertical
    for (int i = 0; i < nrows-1; i++){
      for (int j = 0; j < ncols; j++){
        addEdgeWeight(boundary_term, i*ncols+j, i*ncols+(j+1));
      }
    }

    __END__;
	return CV_StsOk;
  }

  inline void maxflow(){m_g->maxflow();}

  void get_segmentation(CvArr * arr)
  {
    CvMat arrheader;
    CvMat * mat = cvGetMat(arr, &arrheader);
    assert(mat->cols == m_img->cols);
    assert(mat->rows == m_img->rows);
    assert(cvGetElemType(mat)==CV_8U);
    
    for (int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols; j++){
        // show target !
        CV_MAT_ELEM(*mat, uchar, i, j)= 
            (m_g->what_segment(i*ncols+j) == GraphType::SOURCE)?0:255;
      }
    }
  }
};



#endif // __CV_EXT_MAXFLOW_H__
