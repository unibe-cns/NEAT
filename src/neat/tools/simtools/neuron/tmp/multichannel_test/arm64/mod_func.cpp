#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _CaDyn_reg(void);
extern void _CaPump_reg(void);
extern void _ICa_HVA_reg(void);
extern void _ICa_LVAst_reg(void);
extern void _IKv3_1_reg(void);
extern void _INaTa_t_reg(void);
extern void _INa_Ta_reg(void);
extern void _IPiecewiseChannel_reg(void);
extern void _ISK_reg(void);
extern void _ISK_E2_reg(void);
extern void _ISKv3_1_reg(void);
extern void _ITestChannel_reg(void);
extern void _ITestChannel2_reg(void);
extern void _Ih_reg(void);
extern void _NMDA_Mg_T_reg(void);
extern void _OUClamp_reg(void);
extern void _OUClamp2_reg(void);
extern void _OUConductance_reg(void);
extern void _OUReversal_reg(void);
extern void _Shunt_reg(void);
extern void _SinClamp_reg(void);
extern void _VecStim_reg(void);
extern void _WNClamp_reg(void);
extern void _conc_ca_reg(void);
extern void _double_exp_AMPA_NMDA_reg(void);
extern void _double_exp_AMPA_NMDA_modified_reg(void);
extern void _double_exp_AMPA_NMDA_modified2_reg(void);
extern void _double_exp_AMPA_NMDA_rescaled_reg(void);
extern void _double_exp_AMPA_NMDA_rescaled_prox_reg(void);
extern void _double_exp_AMPA_NMDA_v_from_vector_reg(void);
extern void _epsc_double_exp_reg(void);
extern void _exp_AMPA_NMDA_reg(void);
extern void _gap_reg(void);
extern void _rel_reg(void);
extern void _release_BMK_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"mech//CaDyn.mod\"");
    fprintf(stderr, " \"mech//CaPump.mod\"");
    fprintf(stderr, " \"mech//ICa_HVA.mod\"");
    fprintf(stderr, " \"mech//ICa_LVAst.mod\"");
    fprintf(stderr, " \"mech//IKv3_1.mod\"");
    fprintf(stderr, " \"mech//INaTa_t.mod\"");
    fprintf(stderr, " \"mech//INa_Ta.mod\"");
    fprintf(stderr, " \"mech//IPiecewiseChannel.mod\"");
    fprintf(stderr, " \"mech//ISK.mod\"");
    fprintf(stderr, " \"mech//ISK_E2.mod\"");
    fprintf(stderr, " \"mech//ISKv3_1.mod\"");
    fprintf(stderr, " \"mech//ITestChannel.mod\"");
    fprintf(stderr, " \"mech//ITestChannel2.mod\"");
    fprintf(stderr, " \"mech//Ih.mod\"");
    fprintf(stderr, " \"mech//NMDA_Mg_T.mod\"");
    fprintf(stderr, " \"mech//OUClamp.mod\"");
    fprintf(stderr, " \"mech//OUClamp2.mod\"");
    fprintf(stderr, " \"mech//OUConductance.mod\"");
    fprintf(stderr, " \"mech//OUReversal.mod\"");
    fprintf(stderr, " \"mech//Shunt.mod\"");
    fprintf(stderr, " \"mech//SinClamp.mod\"");
    fprintf(stderr, " \"mech//VecStim.mod\"");
    fprintf(stderr, " \"mech//WNClamp.mod\"");
    fprintf(stderr, " \"mech//conc_ca.mod\"");
    fprintf(stderr, " \"mech//double_exp_AMPA_NMDA.mod\"");
    fprintf(stderr, " \"mech//double_exp_AMPA_NMDA_modified.mod\"");
    fprintf(stderr, " \"mech//double_exp_AMPA_NMDA_modified2.mod\"");
    fprintf(stderr, " \"mech//double_exp_AMPA_NMDA_rescaled.mod\"");
    fprintf(stderr, " \"mech//double_exp_AMPA_NMDA_rescaled_prox.mod\"");
    fprintf(stderr, " \"mech//double_exp_AMPA_NMDA_v_from_vector.mod\"");
    fprintf(stderr, " \"mech//epsc_double_exp.mod\"");
    fprintf(stderr, " \"mech//exp_AMPA_NMDA.mod\"");
    fprintf(stderr, " \"mech//gap.mod\"");
    fprintf(stderr, " \"mech//rel.mod\"");
    fprintf(stderr, " \"mech//release_BMK.mod\"");
    fprintf(stderr, "\n");
  }
  _CaDyn_reg();
  _CaPump_reg();
  _ICa_HVA_reg();
  _ICa_LVAst_reg();
  _IKv3_1_reg();
  _INaTa_t_reg();
  _INa_Ta_reg();
  _IPiecewiseChannel_reg();
  _ISK_reg();
  _ISK_E2_reg();
  _ISKv3_1_reg();
  _ITestChannel_reg();
  _ITestChannel2_reg();
  _Ih_reg();
  _NMDA_Mg_T_reg();
  _OUClamp_reg();
  _OUClamp2_reg();
  _OUConductance_reg();
  _OUReversal_reg();
  _Shunt_reg();
  _SinClamp_reg();
  _VecStim_reg();
  _WNClamp_reg();
  _conc_ca_reg();
  _double_exp_AMPA_NMDA_reg();
  _double_exp_AMPA_NMDA_modified_reg();
  _double_exp_AMPA_NMDA_modified2_reg();
  _double_exp_AMPA_NMDA_rescaled_reg();
  _double_exp_AMPA_NMDA_rescaled_prox_reg();
  _double_exp_AMPA_NMDA_v_from_vector_reg();
  _epsc_double_exp_reg();
  _exp_AMPA_NMDA_reg();
  _gap_reg();
  _rel_reg();
  _release_BMK_reg();
}

#if defined(__cplusplus)
}
#endif
