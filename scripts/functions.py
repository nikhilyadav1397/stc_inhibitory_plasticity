import brian2 as b
import numpy as np
from brian2.units import *


# get_model
# Defining the mathematical equations describing the network model
# - req: string specifying which model is wanted
# - return: the requested equation (or set of equations)
def get_model(req):

    # neuron dynamics: LIF membrane potential and amount of plasticity-related proteins (for excitatory neurons)
    if req == "neuron_full":
        return """
        dV/dt = ( -(V - V_rev) + V_psp + R_mem*I_bg + V_stim_0 + V_stim_1 + V_stim_2 + V_stim_3 ) / tau_mem: volt (unless refractory)
        dI_bg/dt = ( -I_bg + I_0 + sigma_wn*xi_bg ) / tau_OU : amp
        V_stim_0 : volt
        V_stim_1 : volt
        V_stim_2 : volt
        V_stim_3 : volt
        sum_h_diff : volt
        nm : 1 (shared)
        theta_pro_c  = 0.001*volt / (nm + 0.001) : volt
        dp/dt = ( -p + alpha_c*0.5*(1+sign(sum_h_diff - theta_pro_c)) ) / tau_pro_c : 1
        dV_psp/dt = -V_psp / tau_syn : volt
        """
    # neuron dynamics: LIF membrane potential (simplified, for inhibitory neurons)
    elif req == "neuron_simple":
        return """
        dV/dt = ( -(V - V_rev) + V_psp + R_mem*I_bg ) / tau_mem: volt (unless refractory)
        dI_bg/dt = ( -I_bg + I_0 + sigma_wn*xi_bg ) / tau_OU : amp
        dV_psp/dt = -V_psp / tau_syn : volt
        """
    elif req == "neuron_consolidation":
        return """
        sum_h_diff : volt
        nm : 1 (shared)
        theta_pro_c  = 0.001*volt / (nm + 0.001) : volt
        dp/dt = ( -p + alpha_c*0.5*(1+sign(sum_h_diff - theta_pro_c)) ) / tau_pro_c : 1
        """
    # plastic synapse dynamics: calcium amount, early-phase weight, tag, late-phase weight
    elif req == "plastic_synapse":
        return """
        dh/dt = ( 0.1*(h_0 - h) + gamma_p*(10*mvolt - h)*0.5*(1+sign(Ca - theta_p)) - gamma_d*h*0.5*(1+sign(Ca - theta_d)) + sqrt(tau_h * (0.5*(1+sign(Ca - theta_p)) + 0.5*(1+sign(Ca - theta_d)))) * sigma_pl * xi_pl ) / tau_h : volt (clock-driven)
        dCa/dt = -Ca/tau_Ca : 1 (clock-driven)
        sum_h_diff_post = abs(h - h_0) : volt (summed)
        dz/dt = ( alpha_c*p_post *(1-z)*0.5*(1+sign((h - h_0) - theta_tag_p)) - alpha_c*p_post *(z + 0.5) * 0.5*(1+sign((h_0 - h) - theta_tag_d)) ) / tau_z : 1 (clock-driven)
        w = h + z*h_0 : volt
        """
    elif req == "plastic_synapse_consolidation":
        return """
        dh/dt = (0.1*(h_0 - h)) / tau_h : volt (clock-driven)
        sum_h_diff_post = abs(h - h_0) : volt (summed)
        dz/dt = (alpha_c*p_post*(1-z)*0.5*(1+sign((h - h_0) - theta_tag_p)) - alpha_c*p_post*(z + 0.5)*0.5*(1+sign((h_0 - h) - theta_tag_d))) / tau_z : 1 (clock-driven)
        w = h + z*h_0 : volt
        """
    # plastic synapse dynamics: calcium amount, early-phase weight
    elif req == "plastic_early_only_synapse":
        return """
        dh/dt = ( 0.1*(h_0 - h) + gamma_p*(10*mvolt - h)*0.5*(1+sign(Ca - theta_p)) - gamma_d*h*0.5*(1+sign(Ca - theta_d)) + sqrt(tau_h * (0.5*(1+sign(Ca - theta_p)) + 0.5*(1+sign(Ca - theta_d)))) * sigma_pl * xi_pl ) / tau_h : volt (clock-driven)
        dCa/dt = -Ca/tau_Ca : 1 (clock-driven)
        sum_h_diff_post = 0*volt : volt (summed)
        dz/dt = 0 / tau_z : 1 (clock-driven)
        w = h : volt
        """
    # plastic synapse dynamics: postsynaptic effects following presynaptic spike
    elif req == "plastic_synapse_pre":
        return {"pre_voltage": "V_psp_post += w", "pre_calcium": "Ca += Ca_pre"}
    # plastic synapse dynamics: postsynaptic effects following postsynaptic spike
    elif req == "plastic_synapse_post":
        return "Ca += Ca_post"
    # Apre: Tracks recent inhibitory presynaptic spikes
    # Apost: Tracks recent inhibitory postsynaptic spikes
    # tau_istdp: Time constant for inhibitory STDP
    # rho_target: Target postsynaptic firing rate for inhibitory STDP
    # alpha_istdp: Scaling factor for inhibitory STDP
    # eta_istdp: Learning rate for inhibitory STDP
    # w_max_istdp: Maximum synaptic weight for inhibitory STDP
    elif req == "inhibitory_stdp_synapse":
        return """
        w : volt

        dApre/dt = -Apre/tau_istdp : 1 (event-driven)
        dApost/dt = -Apost/tau_istdp : 1 (event-driven)

        tau_istdp : second (shared)
        rho_target : Hz (shared)
        alpha_istdp = 2*rho_target*tau_istdp : 1

        eta_istdp : volt (shared)
        w_max_istdp : volt (shared)
        plasticity_on : 1 (shared)
        """
    elif req == "inhibitory_stdp_pre":
        return {"pre_voltage": """
                V_psp_post += w
                Apre += 1
                w = clip(w - plasticity_on*eta_istdp*(Apost - alpha_istdp), -w_max_istdp, 0*volt)
            """}
    elif req == "inhibitory_stdp_post":
        return """
            Apost += 1
            w = clip(w - plasticity_on*eta_istdp*Apre, -w_max_istdp, 0*volt)
        """
    # non-plastic synapse dynamics: weight
    elif req == "static_synapse":
        return """
        w : volt
        """
    # non-plastic synapse dynamics: postsynaptic potential following presynaptic spike
    elif req == "static_synapse_pre":
        return {
            "pre_voltage": "V_psp_post += w",
        }
    # stimulus input #0 for learning pattern 1
    elif req == "stim_input_0":
        return """
        V_stim_0_post = V_stim_pre : volt (summed)
        """
    # stimulus input #1 for learning pattern 2
    elif req == "stim_input_1":
        return """
        V_stim_1_post = V_stim_pre : volt (summed)
        """
    # stimulus input #2 for recall pattern 1
    elif req == "stim_input_2":
        return """
        V_stim_2_post = V_stim_pre : volt (summed)
        """
    # stimulus input #3 for recall pattern 2
    elif req == "stim_input_3":
        return """
        V_stim_3_post = V_stim_pre : volt (summed)
        """
    # unknown specifier
    raise ValueError(f"Unknown model specifier: {req}")


def set_background_current(bg_prot):
    if bg_prot["scheme"] == "FULL":
        I_0 = bg_prot["I_0"] * namp  # 0.15*namp
        sigma_wn = (
            bg_prot["sigma_WN"] * namp * second ** (1 / 2)
        )  # 0.05*namp*second**(1/2)
    else:
        I_0 = 0 * namp
        sigma_wn = 0 * namp * second ** (1 / 2)
    return I_0, sigma_wn


# get_ou_stimulus
# Defining the mathematical description of an Ornstein-Uhlenbeck stimulus input
# - req: string specifying which paradigm is wanted
# - return: the requested equation or set of equations; or `None` if unknown paradigm has been specified
# This is used to define a NeuronGroup
def get_ou_stimulus(req):

    # modeling putative population input for the full duration
    if req == "FULL":
        return """
			dV/dt = ( -V + (N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim)*second*h_0 ) / tau_OU : volt
			V_stim = V : volt
			f_stim : hertz
			N_stim : 1
			"""
    # modeling putative population input with pulse triplet
    elif req == "TRIPLET":
        return """
			dV/dt = ( -V + ((N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim) * 1*second*h_0 ) ) / tau_OU : volt
			V_stim = ( int(t>=t_start) * int(t<=t_start + 0.1*second)
						+ int(t>=t_start + 0.5*second) * int(t<=t_start + 0.6*second)
						+ int(t>=t_start + 1.0*second) * int(t<=t_start + 1.1*second) ) * V : volt
			f_stim : hertz
			N_stim : 1
			t_start : second
			"""
    # modeling putative population input with one pulse of 0.1 s duration
    elif req == "ONEPULSE":
        return """
			dV/dt = ( -V + ((N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim) * 1*second*h_0 ) ) / tau_OU : volt
			V_stim = ( int(t>=t_start) * int(t<=t_start + 0.1*second) ) * V : volt
			f_stim : hertz
			N_stim : 1
			t_start : second
			"""
    return None


def make_stimulus_group(size, protocol, t_start, h_0):
    ou_stimulus = get_ou_stimulus(protocol["scheme"])
    group = b.NeuronGroup(size, ou_stimulus, method="heun")
    group.f_stim = protocol["freq"] * b.hertz
    group.N_stim = protocol["N_stim"]

    if hasattr(group, "t_start"):
        group.t_start = t_start * b.second

    group.V = "N_stim*f_stim*1*second*h_0"
    return group


def update_modulator_with_q(Q, nm_min=0.01, nm_max=0.1, q_mid=0.05, slope=100):
    if Q <= 0:
        return nm_min
    sig = 1.0 / (1.0 + np.exp(-slope * (Q - q_mid)))
    return nm_min + (nm_max - nm_min) * sig
