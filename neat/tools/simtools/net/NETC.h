#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <complex>
#include <tuple>
#include <numeric>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>

#include "Synapses.h"
#include "Ionchannels.h"
#include "Tools.cc"

using namespace std;

struct IODat{
    // diagonal element of matrix
    double denom;
    // for inversion
    double g_val;
    double f_val;
    // for linear term inversion
    double lf_val;
    double lg_val;
};
struct IOLinDat{
    // for inversion with linear terms
    double lg;
    double lf;
};


class NETNode{
private:
    // propagators exponentials
    vector< complex< double > > m_p0, m_p1, m_p2;
    // state variables convolution
    vector< complex< double > > m_yc;
    // time step
    double m_dt;
    // integration mode flag
    int m_integ_mode; // '0' for steady-state inversion, '1' for implicit
                      // convolution and '2' for single exponential integration

    // for inversion
    double m_xx = 0.0, m_yy = 0.0;
    double m_lxx = 0.0, m_lyy = 0.0;
    double m_gg, m_ff;
    double m_lg, m_lf;
    // denominator of soma node
    double m_soma_denom = 1.0;

public:
    // tree structure indices
    int m_index;
    int m_parent_index;
    vector< int > m_child_indices;
    // associated location
    vector< int > m_loc_indices;
    vector< int > m_newloc_indices;
    // voltage variable
    double m_v_node;
    // integration kernel constants
    vector< complex< double > > m_alphas;
    vector< complex< double > > m_gammas;
    // integration kernel single exponential
    double m_alpha;
    double m_gamma;
    // passage counter
    int m_n_passed;
    // diagonal matrix element
    double m_denom;
    // kernel integral (if ''integ_mode'' is '0', kernel value first time step
    // (if integ_mode is '1')
    double m_kbar;
    // flag to indicate whether node integrates soma or not ('0' means no, '1'
    // means yes and '2' means there are no linear terms)
    int m_soma_flag = 0;

    // constructor, destructor
    NETNode();
    ~NETNode();
    // initialization
    void setSomaFlag(bool lin_terms);
    void setSimConstants(double dt, int integ_mode);
    void setSimConstants();
    void reset();
    // inversion
    inline void gatherInput(double xx, double yy);
    inline void gatherInput(IODat in);
    inline void gatherInputLin(double lxx, double lyy);
    inline void multiplyToDenom(double denom);
    inline IODat IO();
    // inline IOLinDat IOLin();
    // inline IOLinDat setLin(double denom);
    inline IOLinDat getLin();
    // inline void resetLin(){m_lxx = 0.0; m_lyy = 0.0;}
    // inline pair< double, double > getLinOut();
    inline double calcV(double v_in, int sign);
    inline double calcV(double v_in);
    inline void calcVLin(double v_in, IOLinDat in);
    inline void pass(){m_n_passed++;}
    // for convolution
    void advance(double dt, double conv_input);
};


class LinTerm{
private:
    // propagators exponentials
    vector< complex< double > > m_p0, m_p1, m_p2;
    // state variables convolution
    vector< complex< double > > m_yc;
    // time step
    double m_dt;
    // integration mode flag
    int m_integ_mode; // '1' for implicit convolution and '2' for single
                      // exponential integration

public:
    // kernel value first time step
    double m_kbar;
    // voltage variable
    double m_v_lin;
    // integration kernel constants
    vector< complex< double > > m_alphas;
    vector< complex< double > > m_gammas;
    // integration kernel single exponential
    double m_alpha;
    double m_gamma;

    // constructor, destructor
    LinTerm(){};
    ~LinTerm(){};
    // initialization
    void setSimConstants(double dt, int integ_mode);
    void setSimConstants();
    void reset();
    // for convolution
    void advance(double dt, double conv_input);

};


class NETSimulator{
private:
    /*
    structural data containers for the NET model
    */
    // number of locations
    int m_n_loc;
    // vector of all nodes (first node should be root)
    vector< NETNode > m_nodes;
    // map of all linear terms (keys are location indices)
    map< int, LinTerm > m_lin_terms;

    /*
    pointer vectors for iteration through the tree graph
    */
    // vector of pointers to nodes that are leafs
    vector< NETNode* > m_leafs;
    // vector for down sweep iteration
    vector< NETNode* > m_down_sweep;
    // vector for up sweep iteration
    vector< NETNode* > m_up_sweep;
    // path from soma leaf to root (for linear layers)
    vector< NETNode* > m_soma_path;

    /*
    to compute and store synaptic input currents
    */
    // vector of all synaptic voltage dependencies
    vector< vector< VoltageDependence* > > m_v_dep;
    // vector of all synaptic conductance windows
    vector< vector< ConductanceWindow* > > m_cond_w;
    // vector of all ionchannels
    vector< vector< IonChannel* > > m_chan;
    // vectors that represent input
    vector< double > m_f_in;
    vector< double > m_df_dv_in;
    // vector of equilibrium potentials
    vector< double > m_v_eq;
    // location voltage vector
    vector< double > m_v_loc;

    /*
    simulation specific parameters
    */
    // timestep
    double m_dt;
    // integration type flag
    double m_integ_mode; // '0' for steady-state inversion, '1' for implicit
                         // convolution and '2' for single exponential integration

    ChannelCreator* m_ccreate = new ChannelCreator();

    //recursion function
    void feedInputs(NETNode* node_ptr);
    void solveMatrixDownSweep(NETNode* node_ptr, vector< NETNode* >::iterator leaf_it,
                                                double& determinant);
    void solveMatrixUpSweep(NETNode& node, double vv, int det_sign);
    void calcLinTerms(NETNode& node, NETNode& pnode);
    void sumV(NETNode& node, double& vv);
    void setVNodeFromVLocUpSweep(NETNode* node, double v_p, double *v_arr);
    // void solveLinDownSweep(NETNode* node, vector< NETNode* >:: iterator leaf_it,
    //                                             IOLinOutput& out);

    // iteration functions
    void setLeafs();
    void setDownSweep();
    void setDownSweep(NETNode* node, vector< NETNode* >:: iterator leaf_it);
    void setUpSweep();
    void setUpSweep(NETNode* node);
    void _getPathToRoot(NETNode* node, vector< NETNode*> &path);

public:
    // constructor, destructor
    NETSimulator(int n_loc, double* v_eq);
    ~NETSimulator();

    // initialization functions from python
    void initFromPython(double dt, double integ_mode, bool print_tree);
    void addNodeFromPython(int node_index, int parent_index,
                            int64_t* child_indices, int n_children,
                            int64_t* loc_indices, int n_locinds,
                            int64_t* newloc_indices, int n_newlocinds,
                            double* alphas, double* gammas, int n_exp);
    void addLinTermFromPython(int loc_index,
                            double* alphas, double* gammas, int n_exp);
    void addIonChannelFromPython(string channel_name, int loc_ind, double g_bar, double e_rev,
                                 bool instantaneous, double* vs, int v_size);
    void addSynapseFromType(int loc_ind, int syn_type);
    void addSynapseFromParams(int loc_ind, double e_r,
                            double *params, int p_size);
    void reset();

    // vectors for iteration
    vector< NETNode* > getPathToRoot(int node_index);
    vector< NETNode* > getPathToRoot(NETNode* node);

    // other structure functions
    void removeSynapseFromIndex(int loc_ind, int syn_ind);
    NETNode* findSomaLeaf();

    // getter functions for voltage and synaptic conductance
    void addVLocToArr(double *v_arr, int v_size);
    vector< double > getVLoc();
    double getVSingleLoc(int loc_index){return m_v_loc[loc_index];};
    void addVNodeToArr(double *v_arr, int v_size);
    vector< double > getVNode();
    double getVSingleNode(int node_index){return m_nodes[node_index].m_v_node;};
    double getGSingleSyn(int loc_index, int syn_index);
    double getSurfaceSingleSyn(int loc_index, int syn_index);

    // integration functions
    // void constructInputs(vector< double > v_m,
    //                         vector< vector< double > > g_syn); // untested
    void constructInputSyn1Loc(int loc_ind, double v_m,
                            double *g_syn, int g_size);
    void constructInputChan1Loc(int loc_ind, double v_m);
    void setInputsToZero();
    void constructMatrix(double dt, double* mat, double* arr, int n_node);
    void solveMatrix();
    void setVNodeFromVLoc(double *v_arr, int v_size);
    void setVNodeFromVNode(double *v_arr, int v_size);
    void advance(double dt);
    void advanceConvolutions(double dt);
    void feedSpike(int loc_ind, int syn_ind, double g_max, int n_spike);
    void record();

    // print functions
    void printTree();
    void printSyns();
};