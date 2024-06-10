#include "NETC.h"


////////////////////////////////////////////////////////////////////////////////
// constructor
NETNode::NETNode(){};
// destructor
NETNode::~NETNode(){};

void NETNode::setSomaFlag(bool lin_terms){
    // set the soma flag
    if(lin_terms){
        if(find(m_loc_indices.begin(), m_loc_indices.end(), 0) != m_loc_indices.end())
            m_soma_flag = 1;
        else
            m_soma_flag = 0;
    } else {
        m_soma_flag = 2;
    }
}

void NETNode::setSimConstants(double dt, int integ_mode){
    // set the simulation constants
    m_integ_mode = integ_mode;
    m_n_passed = 0;
    if(integ_mode == 0){
        complex< double > kbar_aux;
        for(int ii = 0; ii < m_gammas.size(); ii++){
            kbar_aux -= m_gammas[ii] / m_alphas[ii];
        }
        m_kbar = real(kbar_aux);
    } else if(integ_mode == 1){
        // number of exponentials in kernel approximation
        int n_exp = m_alphas.size();
        m_yc.resize(n_exp);
        m_p0.reserve(n_exp);
        m_p1.reserve(n_exp);
        m_p2.reserve(n_exp);
        // initialize the propagators
        complex< double > kbar_aux;
        complex< double > one_c(1.0, 0.0);
        for(int ii = 0; ii < n_exp; ii++){
            m_p0.push_back(exp(m_alphas[ii]*dt));
            m_p1.push_back((m_p0[ii] - one_c) / m_alphas[ii]);
            m_p2.push_back(m_gammas[ii] * m_p0[ii]);
            kbar_aux += m_gammas[ii] * m_p1[ii];
        }
        m_kbar = real(kbar_aux);
        m_dt = dt;
    } else if(integ_mode == 2){
        // TODO;
    } else {
        cerr << "invalid integration mode, should be '0' for steady state, "
                "'1' for implicit convolution and '2' for single exponential";
    }
};
void NETNode::setSimConstants(){
    setSimConstants(0.0, 0);
};

void NETNode::reset(){
    m_v_node = 0.0;
    complex< double > yy(0., 0.);
    fill(m_yc.begin(), m_yc.end(), yy);
}

inline void NETNode::gatherInput(double xx, double yy){
    m_xx += xx; m_yy += yy;
}
inline void NETNode::gatherInput(IODat in){
    m_xx += in.g_val; m_yy += in.f_val;
    if(m_soma_flag == 0 || m_soma_flag == 1){
        m_lxx += in.lg_val; m_lyy += in.lf_val;
    }
}
inline void NETNode::gatherInputLin(double lxx, double lyy){
    m_lxx += lxx; m_lyy += lyy;
}
inline void NETNode::multiplyToDenom(double denom){
    m_soma_denom *= denom;
}

inline IODat NETNode::IO(){
    IODat out;
    // store values linear part
    if(m_soma_flag == 1){
        // do additional stuff since this node is soma and receives linear terms
        // m_soma_denom *= m_denom;
        m_lg = m_lxx;
        m_lf = m_lyy;
        m_xx += m_lg / m_soma_denom; // divide by denom
        m_yy += m_lf / m_soma_denom; // divide by denom
        out.lg_val = 0.0; out.lf_val = 0.0;
    }
    // diagonal matrix element
    m_denom = 1. + m_kbar * m_xx;
    // linear denominators
    if(m_soma_flag == 1){
        m_soma_denom *= m_denom;
        out.denom = m_soma_denom;
    } else if(m_integ_mode == 0){
        out.denom = m_denom;
    }
    // output values
    out.g_val = m_xx / m_denom;
    out.f_val = (m_yy - m_v_node * m_xx) / m_denom;
    // stored values for up sweep
    m_gg = m_kbar * out.g_val;
    m_ff = (m_v_node + m_kbar * m_yy) / m_denom;
    // reset node voltage for up sweep
    if(m_integ_mode != 0)
        m_v_node = 0.0;
    // linear io part
    if(m_soma_flag == 0){
        // normal linear term output
        out.lg_val = (1.0 - m_gg) * m_lxx;
        out.lf_val = m_lyy - m_lxx * m_ff;
    }
    return out;
};

// up sweep for Newton optimization
inline double NETNode::calcV(double v_in, int sign){
    // reset recursion variables
    m_xx = 0.0; m_yy = 0.0;
    // compute voltage
    double dv = m_ff - m_gg * v_in;
    m_v_node += sign * dv;
    return v_in + dv;
};
// up sweep for simulation without linear terms
inline double NETNode::calcV(double v_in){
    // reset recursion variables
    m_xx = 0.0; m_yy = 0.0;
    if(m_soma_flag == 0 || m_soma_flag == 1){
        m_lxx = 0.0; m_lyy = 0.0;
        m_soma_denom = 1.0;
    }
    // compute voltage
    m_v_node += m_ff - m_gg * v_in;
    return m_v_node + v_in;
};

// up sweep with linear terms
inline void NETNode::calcVLin(double v_in, IOLinDat in){
    // compute voltage
    m_v_node += m_kbar * (in.lf - v_in * in.lg) / m_soma_denom;
};

inline IOLinDat NETNode::getLin(){
    IOLinDat ld;
    ld.lg = m_lg; ld.lf = m_lf;
    return ld;
}

inline void NETNode::advance(double dt, double conv_input){
    if(abs(dt - m_dt) > 1e-9) setSimConstants(dt, 1);
    complex< double > v_aux(0.0, 0.0);
    for(int ii = 0; ii < m_yc.size(); ii++){
        m_yc[ii] *= m_p0[ii];
        m_yc[ii] += m_p1[ii] * conv_input;
        v_aux += m_yc[ii] * m_p2[ii];
    }
    m_v_node = real(v_aux);
};
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void LinTerm::setSimConstants(double dt, int integ_mode){
    m_integ_mode = integ_mode;
    if(integ_mode == 1){
        // number of exponentials in kernel approximation
        int n_exp = m_alphas.size();
        m_yc.resize(n_exp);
        m_p0.reserve(n_exp);
        m_p1.reserve(n_exp);
        m_p2.reserve(n_exp);
        // initialize the propagators
        complex< double > kbar_aux;
        complex< double > one_c(1.0, 0.0);
        for(int ii = 0; ii < n_exp; ii++){
            m_p0.push_back(exp(m_alphas[ii]*dt));
            m_p1.push_back((m_p0[ii] - one_c) / m_alphas[ii]);
            m_p2.push_back(m_gammas[ii] * m_p0[ii]);
            kbar_aux += m_gammas[ii] * m_p1[ii];
        }
        m_kbar = real(kbar_aux);
        m_dt = dt;
    } else if(integ_mode == 2){
        // TODO;
    } else {
        cerr << "invalid integration mode, should be "
                "'1' for implicit convolution and '2' for single exponential";
    }
};

void LinTerm::reset(){
    m_v_lin = 0.0;
    complex< double > yy(0., 0.);
    fill(m_yc.begin(), m_yc.end(), yy);
}

void LinTerm::advance(double dt, double conv_input){
    if(abs(dt - m_dt) > 1e-9) setSimConstants(dt, 1);
    complex< double > v_aux(0.0, 0.0);
    for(int ii = 0; ii < m_yc.size(); ii++){
        m_yc[ii] *= m_p0[ii];
        m_yc[ii] += m_p1[ii] * conv_input;
        v_aux += m_yc[ii] * m_p2[ii];
    }
    m_v_lin = real(v_aux);
};
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// constructor
NETSimulator::NETSimulator(int n_loc, double* v_eq):
            m_v_dep(n_loc), m_cond_w(n_loc), m_chan(n_loc),
            m_f_in(n_loc), m_df_dv_in(n_loc),
            m_v_eq(n_loc), m_v_loc(n_loc){
    m_n_loc = n_loc;
    for(int ii=0; ii < n_loc; ii++){
        m_v_eq[ii] = v_eq[ii];
    }
    // fill(m_v_eq.begin(), m_v_eq.end(), v_eq);
};

// destructor
NETSimulator::~NETSimulator(){};

void NETSimulator::initFromPython(double dt, double integ_mode, bool print_tree){
    if(print_tree)
        printTree();
    m_dt = dt;
    m_integ_mode = integ_mode;
    // initialize the nodes
    for(int ii=0; ii<m_nodes.size(); ii++){
        NETNode &node = m_nodes[ii];
        node.setSimConstants(m_dt, m_integ_mode);
        node.setSomaFlag(!m_lin_terms.empty());
    }
    if(m_integ_mode != 0){
        for(auto lt_it = m_lin_terms.begin(); lt_it != m_lin_terms.end(); lt_it++){
            lt_it->second.setSimConstants(m_dt, m_integ_mode);
        }
    }
    // always initialize in this order
    set_leafs();
    setDownSweep();
    setUpSweep();
    // initialize ion channels
    for(int loc_ind = 0; loc_ind < m_n_loc; loc_ind++){
        // advance ion channel currents at locations
        for(vector< IonChannel* >::iterator ioncptr_it = m_chan[loc_ind].begin();
            ioncptr_it != m_chan[loc_ind].end(); ioncptr_it++){
            (*ioncptr_it)->setPOpenEQ(m_v_eq[loc_ind]);
        }
    }
};

void NETSimulator::addNodeFromPython(int node_index, int parent_index,
                                     int64_t* child_indices, int n_children,
                                     int64_t* loc_indices, int n_locinds,
                                     int64_t* newloc_indices, int n_newlocinds,
                                     double* alphas, double* gammas, int n_exp){
    /*
    Add a node to the tree structure via the pyhthon interface
    leafs should have [-1] as child indices
    root shoud have -1 as parent index
    */
    NETNode node;
    node.m_index = node_index;
    node.m_parent_index = parent_index;
    arr2vec(node.m_child_indices, child_indices, n_children);
    arr2vec(node.m_loc_indices, loc_indices, n_locinds);
    arr2vec(node.m_newloc_indices, newloc_indices, n_newlocinds);
    if(node.m_newloc_indices[0] == -1) node.m_newloc_indices.clear();
    arr2vec(node.m_alphas, alphas, n_exp);
    arr2vec(node.m_gammas, gammas, n_exp);
    m_nodes.push_back(node);
};

void NETSimulator::addLinTermFromPython(int loc_index,
                                        double* alphas, double* gammas, int n_exp){
    if(loc_index < 0 || loc_index > m_n_loc) cerr << "'loc_index' out of range" << endl;
    LinTerm lin_term;
    arr2vec(lin_term.m_alphas, alphas, n_exp);
    arr2vec(lin_term.m_gammas, gammas, n_exp);
    m_lin_terms.insert(pair< int, LinTerm >(loc_index, lin_term));
}

void NETSimulator::addIonChannelFromPython(string channel_name, int loc_ind, double g_bar, double e_rev,
                                           bool instantaneous, double* vs, int v_size){
    if(loc_ind < 0 || loc_ind > m_n_loc) cerr << "'loc_ind' out of range" << endl;
    if(g_bar < 0.) cerr << "'g_bar' must be positive" << endl;
    // create the ion channel
    IonChannel* chan = m_ccreate->createInstance(channel_name);
    chan->init(g_bar, e_rev);
    chan->setPOpenEQ(m_v_eq[loc_ind]);
    chan->setInstantaneous(instantaneous);
    chan->setfNewtonConstant(vs, v_size);
    m_chan[loc_ind].push_back(chan);
};

void NETSimulator::addSynapseFromType(int loc_ind, int syn_type){
    if(loc_ind < 0 || loc_ind > m_n_loc) cerr << "'loc_ind' out of range" << endl;
    if(syn_type == 0){
        DrivingForce* syn = new DrivingForce(0.0);
        m_v_dep[loc_ind].push_back(syn);
        ConductanceWindow* cond = new Exp2Cond();
        cond->setParams(0.2, 3.0);
        m_cond_w[loc_ind].push_back(cond);
    } else if(syn_type == 1){
        NMDA* syn = new NMDA(0.0);
        m_v_dep[loc_ind].push_back(syn);
        ConductanceWindow* cond = new Exp2Cond();
        cond->setParams(0.2, 43.0);
        m_cond_w[loc_ind].push_back(cond);
    } else if(syn_type == 2){
        DrivingForce* syn = new DrivingForce(-80.0);
        m_v_dep[loc_ind].push_back(syn);
        ConductanceWindow* cond = new Exp2Cond();
        cond->setParams(0.2, 10.0);
        m_cond_w[loc_ind].push_back(cond);
    } else {
        cerr << "input arg [syn_type] has incorrect value, choose '0' for AMPA, "
                    "'1' for NMDA or '2' for GABA" << endl;
    }
};
void NETSimulator::addSynapseFromParams(int loc_ind, double e_r,
                                            double *params, int p_size){
    if(loc_ind < 0 || loc_ind > m_n_loc) cerr << "'loc_ind out of range" << endl;
    DrivingForce* syn = new DrivingForce(e_r);
    m_v_dep[loc_ind].push_back(syn);
    if(p_size == 1){
        ConductanceWindow* cond = new ExpCond();
        cond->setParams(params[0]);
        m_cond_w[loc_ind].push_back(cond);
    } else if(p_size == 2){
        ConductanceWindow* cond = new Exp2Cond();
        cond->setParams(params[0], params[1]);
        m_cond_w[loc_ind].push_back(cond);
    } else{
        cerr << "size of 'params' should be 1 for single exp window or 2 for "
                    "double exp window" << endl;
    }
}

void NETSimulator::removeSynapseFromIndex(int loc_ind, int syn_ind){
    if(loc_ind < 0 || loc_ind > m_n_loc)
        cerr << "'loc_ind' out of range" << endl;
    if(syn_ind < 0 || syn_ind > (int)m_v_dep[loc_ind].size())
        cerr << "'syn_ind' out of range" << endl;
    VoltageDependence* v_dep_ptr = m_v_dep[loc_ind][syn_ind];
    m_v_dep[loc_ind].erase(m_v_dep[loc_ind].begin() + syn_ind);
    delete v_dep_ptr;
    ConductanceWindow* cond_w_ptr = m_cond_w[loc_ind][syn_ind];
    m_cond_w[loc_ind].erase(m_cond_w[loc_ind].begin() + syn_ind);
    delete cond_w_ptr;
};

void NETSimulator::reset(){
    for(int loc_ind = 0; loc_ind < m_n_loc; loc_ind++){
        for(vector< ConductanceWindow* >::iterator cw_it = m_cond_w[loc_ind].begin();
            cw_it != m_cond_w[loc_ind].end(); cw_it++){
            (*cw_it)->reset();
        }
    }
    for(auto node_it = m_nodes.begin(); node_it != m_nodes.end(); node_it++){
        node_it->reset();
    }
    for(auto lt_it = m_lin_terms.begin(); lt_it != m_lin_terms.end(); lt_it++){
        lt_it->second.reset();
    }

}

void NETSimulator::addVLocToArr(double *v_arr, int v_size){
    if(v_size != int(m_n_loc)) cerr << "'v_arr' has wrong size" << endl;
    // set the array elements to equilibrium potential
    for(int ii = 0; ii < m_n_loc; ii++)
        v_arr[ii] = m_v_eq[ii];
    // set the voltage values
    for(vector< NETNode >::iterator node_it = m_nodes.begin();
        node_it != m_nodes.end(); node_it++){
        for(vector< int >::iterator jj = node_it->m_loc_indices.begin();
            jj != node_it->m_loc_indices.end(); jj++){
            v_arr[*jj] += node_it->m_v_node;
        }
    }
    // add the linear voltage values
    for(auto lt_it = m_lin_terms.begin(); lt_it != m_lin_terms.end(); lt_it++){
        v_arr[0] += lt_it->second.m_v_lin;
    }
};
vector< double > NETSimulator::getVLoc(){
    vector< double > v_loc(m_n_loc);
    addVLocToArr(&v_loc[0], m_n_loc);
    return v_loc;
};

void NETSimulator::addVNodeToArr(double *v_arr, int v_size){
    if(v_size != int(m_nodes.size())) cerr << "'v_arr' has wrong size" << endl;
    for(int ii = 0; ii < m_nodes.size(); ii++){
        v_arr[ii] += m_nodes[ii].m_v_node;
    }
};
vector< double > NETSimulator::getVNode(){
    int n_nodes = int(m_nodes.size());
    vector< double > v_node(n_nodes, 0.0);
    addVNodeToArr(&v_node[0], n_nodes);
    return v_node;
};

double NETSimulator::getGSingleSyn(int loc_index, int syn_index){
    return m_cond_w[loc_index][syn_index]->getCond();
};

double NETSimulator::getSurfaceSingleSyn(int loc_index, int syn_index){
    return m_cond_w[loc_index][syn_index]->getSurface();
};

void NETSimulator::setVNodeFromVLoc(double *v_arr, int v_size){
    if(v_size != m_n_loc) cerr << "'v_arr' has wrong size" << endl;
    vector< double > v_vec;
    arr2vec(v_vec, v_arr, v_size);
    setVNodeFromVLocUpSweep(&m_nodes[0], 0.0, v_arr);
};
void NETSimulator::setVNodeFromVLocUpSweep(NETNode* node_ptr, double v_p,
                                double *v_arr){
    // compute the voltage at current node
    double v_aux = 0.0;
    for(vector< int >::iterator ii = node_ptr->m_newloc_indices.begin();
        ii != node_ptr->m_newloc_indices.end(); ii++){
        v_aux += (v_arr[*ii] - m_v_eq[*ii]);
    }
    if(node_ptr->m_newloc_indices.size() > 0.0){
        v_aux /= double(node_ptr->m_newloc_indices.size());
    };
    node_ptr->m_v_node = v_aux - v_p;
    v_p += node_ptr->m_v_node;
    // go on onto child nodes
    for(vector< int >:: iterator ii = node_ptr->m_child_indices.begin();
        ii != node_ptr->m_child_indices.end(); ii++){
        if(*ii != -1)
            setVNodeFromVLocUpSweep(&m_nodes[*ii], v_p, v_arr);
    }
}

void NETSimulator::setVNodeFromVNode(double *v_arr, int v_size){
    if (v_size != int(m_nodes.size())) cerr << "'v_arr' has wrong size" << endl;
    for(vector< NETNode >::iterator node_it = m_nodes.begin();
        node_it != m_nodes.end(); node_it++){
        node_it->m_v_node = v_arr[node_it - m_nodes.begin()];
    }
}

void NETSimulator::set_leafs(){
    m_leafs.clear();
    for(vector< NETNode >::iterator node_it = m_nodes.begin();
        node_it != m_nodes.end(); node_it++){
        if((*node_it).m_child_indices[0] == -1){
            m_leafs.push_back(&(*node_it));
        }
    }
}

void NETSimulator::setDownSweep(){
    m_down_sweep.clear();
    vector< NETNode* >::iterator leaf_it = m_leafs.begin();
    setDownSweep(m_leafs[0], leaf_it);
}
void NETSimulator::setDownSweep(NETNode* node,
                             vector< NETNode* >:: iterator leaf_it){
    // compute the input output transformation at node
    m_down_sweep.push_back(node);
    // move on the parent layer
    if(node->m_parent_index != -1){
        NETNode* pnode = &m_nodes[node->m_parent_index];
        // move on to next nodes
        pnode->m_n_passed++;
        if(pnode->m_n_passed == int(pnode->m_child_indices.size())){
            pnode->m_n_passed = 0;
            // move on to next node
            setDownSweep(pnode, leaf_it);
        } else {
            // start at next leaf
            leaf_it++;
            if(leaf_it != m_leafs.end())
                setDownSweep(*leaf_it, leaf_it);
        }
    }
}

void NETSimulator::setUpSweep(){
    m_up_sweep.clear();
    setUpSweep(m_down_sweep.back());
}
void NETSimulator::setUpSweep(NETNode* node){
    m_up_sweep.push_back(node);
    // move on to child nodes
    for(vector< int >::iterator ii = node->m_child_indices.begin();
        ii != node->m_child_indices.end(); ii++){
        if(*ii != -1)
            setUpSweep(&m_nodes[*ii]);
    }
}

vector< NETNode* > NETSimulator::getPathToRoot(int node_index){
    vector< NETNode* > path;
    _getPathToRoot(&m_nodes[node_index], path);
    return path;
}
vector< NETNode* > NETSimulator::getPathToRoot(NETNode* node){
    vector< NETNode* > path;
    _getPathToRoot(node, path);
    return path;
}
void NETSimulator::_getPathToRoot(NETNode* node, vector< NETNode* > &path){
    path.push_back(node);
    if(node->m_parent_index != -1)
        _getPathToRoot(&m_nodes[node->m_parent_index], path);
}

// void NETSimulator::constructInputs(vector< double > v_m,
//                                    vector< vector< double > > g_syn){
//     // check sizes
//     size_t n_loc = m_n_loc;
//     if(v_m.size() != n_loc) std::cerr << "v_m has wrong size" << endl;
//     if(g_syn.size() != n_loc) std::cerr << "g_syn has wrong size" << endl;
//     for(int ii = 0; ii < g_syn.size(); ii++){
//         if(g_syn[ii].size() != m_v_dep[ii].size()){
//             cerr << "g_syn has wrong size" << endl;
//         }
//     }
//     // construct the inputs
//     setInputsToZero();
//     for(int ii = 0; ii < m_n_loc; ii++){
//         // synapse
//         int n_syn = int(g_syn.size());
//         if(n_syn > 0)
//             constructInputSyn1Loc(ii, v_m[ii], &g_syn[ii][0], n_syn);
//         // ion channels
//         int n_chan = int(m_chan.size());
//         if(n_chan > 0)
//             constructInputChan1Loc(ii, v_m[ii]);
//     }
// }

void NETSimulator::setInputsToZero(){
    fill(m_f_in.begin(), m_f_in.end(), 0.0);
    fill(m_df_dv_in.begin(), m_df_dv_in.end(), 0.0);
}

void NETSimulator::constructInputSyn1Loc(int loc_ind, double v_m,
                                      double *g_syn, int g_size){
    for(int jj = 0; jj < g_size; jj++){
        m_f_in[loc_ind] -= g_syn[jj] * m_v_dep[loc_ind][jj]->f(v_m);
        m_df_dv_in[loc_ind] -= g_syn[jj] * m_v_dep[loc_ind][jj]->DfDv(v_m);
    }
}

void NETSimulator::constructInputChan1Loc(int loc_ind, double v_m){
    // construct aglrotihm input values at current location for channel
    for(int jj = 0; jj < m_chan[loc_ind].size(); jj++){
        m_f_in[loc_ind] -= m_chan[loc_ind][jj]->getCondNewton() *
                           m_chan[loc_ind][jj]->fNewton(v_m);
        m_df_dv_in[loc_ind] -= m_chan[loc_ind][jj]->getCondNewton() *
                               m_chan[loc_ind][jj]->DfDvNewton(v_m);
    }
}

void NETSimulator::constructMatrix(double dt,
                                   double* mat, double* vec, int n_node){
    if(n_node != (int)m_nodes.size()){cerr << "input size wrong!" << endl;}
    // construct vector with nodes with new loc indices
    vector< NETNode* > leafs;
    for(auto node_it = m_nodes.begin(); node_it != m_nodes.end(); node_it++){
        if((int)node_it->m_newloc_indices.size() > 0)
            leafs.push_back(&(*node_it));
    }
    // advance the convolutions for each node (should be made recursive!!!)
    // advanceConvolutions(dt);
    // set diagonal elements to one
    for(int ii = 0; ii < n_node; ii++) mat[ii*n_node + ii] = 1.0;
    // get path for linear terms
    NETNode* soma_leaf_ptr = findSomaLeaf();
    vector< NETNode* > path0 = getPathToRoot(soma_leaf_ptr);
    // construct the normal matrix elements
    for(auto leaf_it = m_leafs.begin(); leaf_it != m_leafs.end(); leaf_it++){
        NETNode* leaf_ptr = *leaf_it;
        vector< NETNode* > path = getPathToRoot(leaf_ptr);
        double gg = 0.0, ff = 0.0;
        double gl = 0.0, fl = 0.0;
        for(auto ii = leaf_ptr->m_newloc_indices.begin();
            ii != leaf_ptr->m_newloc_indices.end(); ii++){
            if(m_integ_mode == 0){
                gg += m_df_dv_in[*ii];
                ff += m_f_in[*ii];
            } else if(m_integ_mode == 1){
                gg += m_df_dv_in[*ii];
                ff += m_df_dv_in[*ii] * (m_v_loc[*ii] - m_v_eq[*ii]) - m_f_in[*ii];
                // add the linear terms
                if(m_lin_terms.find(*ii) != m_lin_terms.end()){
                    double g_lin = m_df_dv_in[0] *
                                   m_lin_terms.at(*ii).m_kbar * m_df_dv_in[*ii];
                    gl += g_lin;
                    fl += g_lin * (m_v_loc[*ii] - m_v_eq[*ii]);
                }
            }
        }
        // add input to nodes on path
        for(auto pn_it0 = path.begin(); pn_it0 != path.end(); pn_it0++){
            NETNode* n0 = *pn_it0;
            // set vector element
            vec[n0->m_index] += n0->m_kbar * ff;
            for(auto pn_it1 = path.begin(); pn_it1 != path.end(); pn_it1++){
                NETNode* n1 = *pn_it1;
                // set matrix element
                mat[n0->m_index*n_node + n1->m_index] += n0->m_kbar * gg;
            }
        }
        // construct linear matrix elements
        for(auto pn_it = path0.begin(); pn_it != path0.end(); pn_it++){
            NETNode* n0 = *pn_it;
            // set vector element
            vec[n0->m_index] += n0->m_kbar * fl;
            for(auto pn_it1 = path.begin(); pn_it1 != path.end(); pn_it1++){
                NETNode* n1 = *pn_it1;
                // set matrix element
                mat[n0->m_index*n_node + n1->m_index] += n0->m_kbar * gl;
            }
        }
    }
}

void NETSimulator::feedInputs(NETNode* node_ptr){
    if(m_integ_mode == 0){
        for(vector< int >:: iterator ii = node_ptr->m_newloc_indices.begin();
            ii != node_ptr->m_newloc_indices.end(); ii++){
                node_ptr->gatherInput(m_df_dv_in[*ii], m_f_in[*ii]);
        }
    } else if(m_integ_mode == 1){
        for(vector< int >:: iterator ii = node_ptr->m_newloc_indices.begin();
            ii != node_ptr->m_newloc_indices.end(); ii++){
            // gather input normal nodes
            node_ptr->gatherInput(m_df_dv_in[*ii],
                        m_df_dv_in[*ii] *
                        (m_v_loc[*ii] - m_v_eq[*ii]) - m_f_in[*ii]);
            // gather inputs for linear transfer
            if(m_lin_terms.find(*ii) != m_lin_terms.end()){
                double g_lin = m_df_dv_in[0] *
                               m_lin_terms.at(*ii).m_kbar * m_df_dv_in[*ii];
                double f_lin = g_lin * (m_v_loc[*ii] - m_v_eq[*ii]);
                node_ptr->gatherInputLin(g_lin, f_lin);
            }
        }
    }
}

// solve matrix with O(n) algorithm
void NETSimulator::solveMatrix(){
    vector< NETNode* >::iterator leaf_it = m_leafs.begin();
    double det = 1.0;
    double& determinant = det;
    // start the down sweep (puts to zero the sub diagonal matrix elements)
    solveMatrixDownSweep(m_leafs[0], leaf_it, determinant);
    // determinant sign
    double det_sign = (determinant < 0.0) - (determinant > 0.0);
    // do up sweep to set voltages
    solveMatrixUpSweep(m_nodes[0], 0.0, det_sign);
}
void NETSimulator::solveMatrixDownSweep(NETNode* node_ptr,
                             vector< NETNode* >::iterator leaf_it,
                             double& determinant){
    // add the new inputs
    feedInputs(node_ptr);
    // compute the input output transformation at node
    IODat output = node_ptr->IO();
    // to stabilize newton iteration
    determinant *= output.denom;
    // move on to the parent layer
    if(node_ptr->m_parent_index != -1){
        NETNode* pnode_ptr = &m_nodes[node_ptr->m_parent_index];
        // gather input from child layers
        pnode_ptr->gatherInput(output);
        // store denominator if necessary for linear terms
        if(m_integ_mode == 1 && !m_lin_terms.empty() && node_ptr->m_soma_flag == 1)
            pnode_ptr->multiplyToDenom(output.denom);
        // move on to next nodes
        pnode_ptr->pass();
        if(pnode_ptr->m_n_passed == int(pnode_ptr->m_child_indices.size())){
            pnode_ptr->m_n_passed = 0;
            // move on to next node
            solveMatrixDownSweep(pnode_ptr, leaf_it, determinant);
        } else {
            // start at next leaf
            leaf_it++;
            if(leaf_it != m_leafs.end())
                solveMatrixDownSweep(*leaf_it, leaf_it, determinant);
        }
    }
}
void NETSimulator::solveMatrixUpSweep(NETNode& node, double vv, int det_sign){
    // compute node voltage
    if(m_integ_mode == 0){
        vv = node.calcV(vv, det_sign);
    } else {
        if(!m_lin_terms.empty() && node.m_soma_flag == 1)
            // add linear terms to node voltage
            calcLinTerms(node, node);
        vv = node.calcV(vv);
    }
    // move on to child nodes
    for(vector< int >:: iterator ii = node.m_child_indices.begin();
        ii != node.m_child_indices.end(); ii++){
        if(*ii != -1)
            solveMatrixUpSweep(m_nodes[*ii], vv, det_sign);
    }
}
void NETSimulator::calcLinTerms(NETNode& node, NETNode& pnode){
    if(pnode.m_parent_index != -1){
        NETNode& pnode_new = m_nodes[pnode.m_parent_index];
        IOLinDat lin_dat = pnode_new.getLin();
        // voltage values of underlying layers
        double v_lin = 0.0;
        sumV(pnode_new, v_lin);
        node.calcVLin(v_lin, lin_dat);
        calcLinTerms(node, pnode_new);
    }
}
void NETSimulator::sumV(NETNode& node, double& vv){
    vv += node.m_v_node;
    if(node.m_parent_index != -1){
        sumV(m_nodes[node.m_parent_index], vv);
    }
}

void NETSimulator::advance(double dt){
    // reset vectors
    fill(m_f_in.begin(), m_f_in.end(), 0.0);
    fill(m_df_dv_in.begin(), m_df_dv_in.end(), 0.0);
    // get the location voltage
    addVLocToArr(&m_v_loc[0], m_v_loc.size());
    // synaptic inputs
    for(int loc_ind = 0; loc_ind < m_n_loc; loc_ind++){
        // advance the synaptic inputs at current location
        for(vector< ConductanceWindow* >::iterator cwptr_it = m_cond_w[loc_ind].begin();
            cwptr_it != m_cond_w[loc_ind].end(); cwptr_it++){
            (*cwptr_it)->advance(dt);
        }
        // advance ion channel currents at locations
        // cout << "DT = " << dt << endl;
        for(vector< IonChannel* >::iterator ioncptr_it = m_chan[loc_ind].begin();
            ioncptr_it != m_chan[loc_ind].end(); ioncptr_it++){
            // cout << "Channel" << endl;
            (*ioncptr_it)->calcFunStatevar(m_v_loc[loc_ind]);
            (*ioncptr_it)->advance(dt);
            (*ioncptr_it)->setPOpen();

        }
        // construct algorithm input values at current location for synapses
        for(int jj = 0; jj < m_cond_w[loc_ind].size(); jj++){
            m_f_in[loc_ind] -= m_cond_w[loc_ind][jj]->getCond() *
                               m_v_dep[loc_ind][jj]->f(m_v_loc[loc_ind]);
            m_df_dv_in[loc_ind] -= m_cond_w[loc_ind][jj]->getCond() *
                                   m_v_dep[loc_ind][jj]->DfDv(m_v_loc[loc_ind]);
        }
        // construct aglrotihm input values at current location for channel
        for(int jj = 0; jj < m_chan[loc_ind].size(); jj++){
            m_f_in[loc_ind] -= m_chan[loc_ind][jj]->getCond() *
                               m_chan[loc_ind][jj]->f(m_v_loc[loc_ind]);
            m_df_dv_in[loc_ind] -= m_chan[loc_ind][jj]->getCond() *
                                   m_chan[loc_ind][jj]->DfDv(m_v_loc[loc_ind]);
        }
    }
    //compute the convolutions at nodes and linear layers
    advanceConvolutions(dt);
    // solve for the next time step's voltage
    solveMatrix();
}

void NETSimulator::advanceConvolutions(double dt){
    // compute linear terms
    for(auto lt_it = m_lin_terms.begin(); lt_it != m_lin_terms.end(); lt_it++){
        lt_it->second.advance(dt, -m_f_in[lt_it->first]);
    }
    // advance the convolutions for each node (should be made recursive!!!)
    for(auto node_it = m_nodes.begin(); node_it != m_nodes.end(); node_it++){
        double conv_input = 0.0;
        for(vector< int >::iterator ii = node_it->m_loc_indices.begin();
            ii != node_it->m_loc_indices.end(); ii++){
            conv_input -= m_f_in[*ii];
        }
        node_it->advance(dt, conv_input);
    }
}

NETNode* NETSimulator::findSomaLeaf(){
    // search for the somatic leaf node
    auto leaf_it = m_leafs.begin();
    while((*leaf_it)->m_loc_indices[0] != 0 && leaf_it != m_leafs.end()){
        leaf_it ++;
    }
    return *leaf_it;
}

void NETSimulator::feedSpike(int loc_ind, int syn_ind, double g_max, int n_spike){
    m_cond_w[loc_ind][syn_ind]->feedSpike(g_max, n_spike);
};

void NETSimulator::printSyns(){
    for(int ii=0; ii < m_v_dep.size(); ii++){
        printf(">>> loc %d --> ", ii);
        for(int jj=0; jj < m_v_dep[ii].size(); jj++){
            printf("synapse type: %d, ", typeid(*m_v_dep[ii][jj]).name());
            printf("e_rev = %.2f mV --- ", m_v_dep[ii][jj]->getEr());
        }
        printf("\n");
    }
    printf("\n");
};

void NETSimulator::printTree(){
    // loop over all nodes
    std::printf(">>> Tree with %d input locations <<<\n", m_n_loc);
    for(int ii=0; ii<m_nodes.size(); ii++){
        NETNode &node = m_nodes[ii];
        cout << "Node " << node.m_index << ", ";
        cout << "Parent node: " << node.m_parent_index << ", ";
        cout << "Child nodes: " << vec2string(node.m_child_indices) << ", ";
        cout << "Location indices: " << vec2string(node.m_loc_indices) << " ";
        cout << "(new: " << vec2string(node.m_newloc_indices) << ")" << endl;
    }
    cout << endl;
};
////////////////////////////////////////////////////////////////////////////////
