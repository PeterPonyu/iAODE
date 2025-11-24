export type AgentParams = {
    later: string;
    batch_percent: number; 
    recon: number;
    irecon: number; 
    beta: number; 
    dip: number;
    tc: number;
    info: number;
    hidden_dims: number;
    latent_dim: number;
    i_dim: number;
    use_ode: boolean;
    loss_mode: 'mse' | 'nb' | 'zinb';
    lr: number;
    vae_reg: number;
    ode_reg: number;
};

export type TrainingParams = {
    epochs: number;
};

export type TrainingState = {
    epoch: number;
    ARI: number;
    NMI: number;
    ASW: number;
    CAL: number;
    DAV: number;
    COR: number;
};

export type DataInfo = {
    n_cells: number;
    n_genes: number;
};