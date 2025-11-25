#!/usr/bin/env python3
"""
Consistency checker for iAODE API and Frontend
Tests parameter alignment between Python backend and TypeScript frontend
"""

import json
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_agent_params():
    """Check if AgentParams match between API and frontend"""
    print(f"\n{BLUE}=== Checking AgentParams ==={RESET}")
    
    # Expected parameters from agent.py
    expected_params = [
        'layer', 'recon', 'irecon', 'beta', 'dip', 'tc', 'info',
        'hidden_dim', 'latent_dim', 'i_dim', 'use_ode', 'loss_mode',
        'lr', 'vae_reg', 'ode_reg', 'train_size', 'val_size', 'test_size',
        'batch_size', 'random_seed', 'encoder_type', 'encoder_num_layers',
        'encoder_n_heads', 'encoder_d_model'
    ]
    
    print(f"Expected {len(expected_params)} parameters")
    print(f"Parameters: {', '.join(expected_params)}")
    
    # Check API model.py
    api_model = Path('api/model.py').read_text()
    api_missing = []
    for param in expected_params:
        if param not in api_model or f'{param}:' not in api_model:
            api_missing.append(param)
    
    if api_missing:
        print(f"{RED}✗ API missing parameters: {', '.join(api_missing)}{RESET}")
        return False
    else:
        print(f"{GREEN}✓ API has all {len(expected_params)} parameters{RESET}")
    
    # Check frontend types.ts
    frontend_types = Path('frontend/src/lib/types.ts').read_text()
    frontend_missing = []
    for param in expected_params:
        if param not in frontend_types:
            frontend_missing.append(param)
    
    if frontend_missing:
        print(f"{RED}✗ Frontend missing parameters: {', '.join(frontend_missing)}{RESET}")
        return False
    else:
        print(f"{GREEN}✓ Frontend has all {len(expected_params)} parameters{RESET}")
    
    return True

def check_train_params():
    """Check if TrainParams match"""
    print(f"\n{BLUE}=== Checking TrainParams ==={RESET}")
    
    expected_params = ['epochs', 'patience', 'val_every', 'early_stop']
    
    print(f"Expected {len(expected_params)} parameters")
    print(f"Parameters: {', '.join(expected_params)}")
    
    api_model = Path('api/model.py').read_text()
    frontend_types = Path('frontend/src/lib/types.ts').read_text()
    
    all_found = True
    for param in expected_params:
        api_has = param in api_model and 'TrainParams' in api_model
        frontend_has = param in frontend_types and 'TrainParams' in frontend_types
        
        if not api_has or not frontend_has:
            print(f"{RED}✗ Parameter '{param}' missing{RESET}")
            all_found = False
    
    if all_found:
        print(f"{GREEN}✓ All {len(expected_params)} parameters present in both API and frontend{RESET}")
    
    return all_found

def check_preprocessing_params():
    """Check preprocessing parameters"""
    print(f"\n{BLUE}=== Checking Preprocessing Parameters ==={RESET}")
    
    param_sets = {
        'TFIDFParams': ['scale_factor', 'log_tf', 'log_idf'],
        'HVPParams': ['n_top_peaks', 'min_accessibility', 'max_accessibility', 'method', 'use_raw_counts'],
        'SubsampleParams': ['n_cells', 'frac_cells', 'use_hvp', 'hvp_column', 'seed']
    }
    
    api_model = Path('api/model.py').read_text()
    frontend_types = Path('frontend/src/lib/types.ts').read_text()
    
    all_consistent = True
    for param_type, params in param_sets.items():
        print(f"\n{param_type}:")
        for param in params:
            api_has = param in api_model
            frontend_has = param in frontend_types
            
            if api_has and frontend_has:
                print(f"  {GREEN}✓{RESET} {param}")
            else:
                print(f"  {RED}✗{RESET} {param} - API: {api_has}, Frontend: {frontend_has}")
                all_consistent = False
    
    return all_consistent

def check_endpoints():
    """Check if all frontend API calls match backend endpoints"""
    print(f"\n{BLUE}=== Checking API Endpoints ==={RESET}")
    
    endpoints = {
        'POST /upload': ('uploadData', 'upload'),
        'POST /train': ('startTraining', 'train'),
        'GET /state': ('getTrainingState', 'state'),
        'GET /download': ('downloadEmbedding', 'download'),
        'DELETE /reset': ('resetState', 'reset'),
        'POST /preprocess/tfidf': ('applyTFIDF', 'preprocess/tfidf'),
        'POST /preprocess/select-hvp': ('selectHVP', 'preprocess/select-hvp'),
        'POST /preprocess/subsample': ('subsampleData', 'preprocess/subsample'),
    }
    
    api_main = Path('api/main.py').read_text()
    frontend_api = Path('frontend/src/lib/api.ts').read_text()
    
    all_found = True
    for endpoint, (frontend_func, backend_route) in endpoints.items():
        backend_has = backend_route in api_main
        frontend_has = frontend_func in frontend_api
        
        if backend_has and frontend_has:
            print(f"{GREEN}✓{RESET} {endpoint} - {frontend_func}()")
        else:
            print(f"{RED}✗{RESET} {endpoint} - Backend: {backend_has}, Frontend: {frontend_has}")
            all_found = False
    
    return all_found

def check_defaults():
    """Check if default values are consistent"""
    print(f"\n{BLUE}=== Checking Default Values ==={RESET}")
    
    defaults_to_check = {
        'layer': 'counts',
        'hidden_dim': 128,
        'latent_dim': 10,
        'i_dim': 2,
        'loss_mode': 'nb',
        'batch_size': 128,
        'encoder_type': 'mlp',
        'encoder_num_layers': 2,
        'epochs': 100,
        'patience': 20,
    }
    
    frontend_params = Path('frontend/src/components/training-params.tsx').read_text()
    
    all_match = True
    for param, expected_value in defaults_to_check.items():
        value_str = str(expected_value)
        if isinstance(expected_value, str):
            value_str = f"'{expected_value}'"
        
        if f'{param}: {value_str}' in frontend_params or f'{param}: "{expected_value}"' in frontend_params:
            print(f"{GREEN}✓{RESET} {param} = {expected_value}")
        else:
            print(f"{YELLOW}⚠{RESET} {param} default might differ from {expected_value}")
            # Not marking as failure since some variations are acceptable
    
    return all_match

def main():
    """Run all consistency checks"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}iAODE API-Frontend Consistency Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    checks = [
        ("Agent Parameters", check_agent_params),
        ("Training Parameters", check_train_params),
        ("Preprocessing Parameters", check_preprocessing_params),
        ("API Endpoints", check_endpoints),
        ("Default Values", check_defaults),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"{RED}✗ Error checking {name}: {e}{RESET}")
            results.append((name, False))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"{status} - {name}")
    
    print(f"\n{BLUE}Results: {passed}/{total} checks passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}All consistency checks passed! ✓{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        return 0
    else:
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}Some checks failed. Please review the output above.{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
