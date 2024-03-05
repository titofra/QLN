#ifndef SETUP_HPP
#define SETUP_HPP

/* INCLUDES */

#include <PNEATM/population.hpp>
#include <PNEATM/genome.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <sndfile.h>
#include <vector>
#include <filesystem>
#include <cfloat>
#include <complex>
#include <fftw3.h>


/* DEFINITIONS */

#define UNUSED(expr) do { (void) (expr); } while (0)

// parameters
const double params_init_extremums = 10.0;
const double params_perturb_extremums = 1.0;
typedef struct activationFnParams {
    double alpha = Random_Double (-params_init_extremums, params_init_extremums);
    double beta = Random_Double (-params_init_extremums, params_init_extremums);
} activationFnParams_t;

// activation functions
std::function<std::complex<double>  (std::complex<double> , activationFnParams_t*)> identity = [] (std::complex<double> x, activationFnParams_t* params) -> std::complex<double>  {
    return x;
    UNUSED (params);
};
std::function<std::complex<double> (std::complex<double>, activationFnParams_t*)> sigmoid_real = [] (std::complex<double> x, activationFnParams_t* params) -> std::complex<double> {
    return std::complex<double> (1.0 / (1.0 + std::exp(params->alpha * (x.real () - params->beta))), x.imag ());
};
std::function<std::complex<double> (std::complex<double>, activationFnParams_t*)> sigmoid_imag = [] (std::complex<double> x, activationFnParams_t* params) -> std::complex<double> {
    return std::complex<double> (x.real (), 1.0 / (1.0 + std::exp(params->alpha * (x.imag () - params->beta))));
};
std::function<std::complex<double> (std::complex<double>, activationFnParams_t*)> lrelu_real = [] (std::complex<double> x, activationFnParams_t* params) -> std::complex<double> {
    if (x.real () > 0.0) {
        return x;
    } else {
        return std::complex<double> (x.real () * params->alpha, x.imag ());
    }
};
std::function<std::complex<double> (std::complex<double>, activationFnParams_t*)> lrelu_imag = [] (std::complex<double> x, activationFnParams_t* params) -> std::complex<double> {
    if (x.imag () > 0.0) {
        return x;
    } else {
        return std::complex<double> (x.real (), x.imag () * params->alpha);
    }
};
std::function<std::complex<double> (std::complex<double>, activationFnParams_t*)> output_dis = [] (std::complex<double> x, activationFnParams_t* params) -> std::complex<double> {
    return std::complex<double> (1.0 - std::exp(-1.0 * std::norm (x)), 0.0);
    UNUSED (params);
};

// mutation functions
std::function<void (activationFnParams_t*, double)> identity_mutation = [] (activationFnParams_t* params, double fitness) -> void {
    UNUSED (params);
    UNUSED (fitness);
};
std::function<void (activationFnParams_t*, double)> sigmoid_mutation = [] (activationFnParams_t* params, double fitness) -> void {
    if (Random_Double (0.0, 1.0, true, false) < 0.3) {
        // reset values
        params->alpha = Random_Double (-params_init_extremums, params_init_extremums);
        params->beta = Random_Double (-params_init_extremums, params_init_extremums);
    } else {
        // perturb values
        params->alpha += params->alpha * Random_Double (-params_perturb_extremums, params_perturb_extremums);
        params->beta += params->beta * Random_Double (-params_perturb_extremums, params_perturb_extremums);
    }
    UNUSED (fitness);
};
std::function<void (activationFnParams_t*, double)> lrelu_mutation = [] (activationFnParams_t* params, double fitness) -> void {
    if (Random_Double (0.0, 1.0, true, false) < 0.3) {
        // reset values
        params->alpha = Random_Double (-params_init_extremums, params_init_extremums);
    } else {
        // perturb values
        params->alpha += params->alpha * Random_Double (-params_perturb_extremums, params_perturb_extremums);
    }
    UNUSED (fitness);
};

// printing functions
std::function<void (activationFnParams_t*, std::string)> noprinting = [] (activationFnParams_t* params, std::string prefix) -> void {
    UNUSED (params);
    UNUSED (prefix);
};
std::function<void (activationFnParams_t*, std::string)> sigmoid_printing = [] (activationFnParams_t* params, std::string prefix) -> void {
    std::cout << prefix << "alpha = " << params->alpha << "\tbeta = " << params->beta;
};
std::function<void (activationFnParams_t*, std::string)> lrelu_printing = [] (activationFnParams_t* params, std::string prefix) -> void {
    std::cout << prefix << "alpha = " << params->alpha;
};


/* SETUP FUNCTIONS */

std::vector<void*> bias_init_gen (2);
std::vector<void*> resetValues_gen (1);
std::vector<ActivationFnBase*> inputsActivationFns_gen;
std::vector<ActivationFnBase*> outputsActivationFns_gen;
std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns_gen;
std::vector<void*> bias_init_dis (2);
std::vector<void*> resetValues_dis (1);
std::vector<ActivationFnBase*> inputsActivationFns_dis;
std::vector<ActivationFnBase*> outputsActivationFns_dis;
std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns_dis;

pneatm::Population<std::complex<double>> SetupPopulation_gen (unsigned int popSize, unsigned int fft_output_sz, unsigned int max_recurrency, spdlog::logger* logger, const std::string& stats_filepath) {
    // nodes scheme setup
    std::vector<size_t> bias_sch (1, 2);
    std::vector<size_t> inputs_sch (1, fft_output_sz);
    std::vector<size_t> outputs_sch (1, fft_output_sz);
    std::vector<std::vector<size_t>> hiddens_sch_init = {{1}};

    // bias values
    std::complex<double>* unit_real = new std::complex<double> (1.0, 0.0);
    std::complex<double>* unit_imag = new std::complex<double> (0.0, 1.0);
    bias_init_gen [0] = (void*) unit_real;
    bias_init_gen [1] = (void*) unit_imag;

    // reset values
    std::complex<double>* nulldouble = new std::complex<double> (0.0, 0.0);
    resetValues_gen [0] = (void*) nulldouble;

    // activation functions setup
    // bias & inputs
    inputsActivationFns_gen = std::vector<ActivationFnBase*> (bias_sch [0] + inputs_sch [0]);
    for (ActivationFnBase*& actfn : inputsActivationFns_gen) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &identity);
        actfn->setPrintingFunction (noprinting);
    }
    // output
    outputsActivationFns_gen = std::vector<ActivationFnBase*> (outputs_sch [0]);
    for (ActivationFnBase*& actfn : outputsActivationFns_gen) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &identity);
        actfn->setPrintingFunction (noprinting);
    }
    // hiddens
    activationFns_gen = std::vector<std::vector<std::vector<ActivationFnBase*>>> (1, std::vector<std::vector<ActivationFnBase*>> (1, std::vector<ActivationFnBase*> (5)));
    activationFns_gen [0][0][0] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][0]->setFunction ((void*) &identity);
    activationFns_gen [0][0][0]->setPrintingFunction (noprinting);
    activationFns_gen [0][0][0]->setMutationFunction (identity_mutation);
    activationFns_gen [0][0][1] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][1]->setFunction ((void*) &sigmoid_real);
    activationFns_gen [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns_gen [0][0][1]->setMutationFunction (sigmoid_mutation);
    activationFns_gen [0][0][2] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][2]->setFunction ((void*) &sigmoid_imag);
    activationFns_gen [0][0][2]->setPrintingFunction (sigmoid_printing);
    activationFns_gen [0][0][2]->setMutationFunction (sigmoid_mutation);
    activationFns_gen [0][0][3] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][3]->setFunction ((void*) &lrelu_real);
    activationFns_gen [0][0][3]->setPrintingFunction (lrelu_printing);
    activationFns_gen [0][0][3]->setMutationFunction (lrelu_mutation);
    activationFns_gen [0][0][4] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][4]->setFunction ((void*) &lrelu_imag);
    activationFns_gen [0][0][4]->setPrintingFunction (lrelu_printing);
    activationFns_gen [0][0][4]->setMutationFunction (lrelu_mutation);

    unsigned int N_ConnInit = 2;
    double probRecuInit = 0.5;
    double weightExtremumInit = 10.0;
    unsigned int maxRecuInit = max_recurrency;
    double speciationThreshInit = 100.0;
    std::vector<genomeStruct_t> specific_genomes = {};
    distanceFn dstType = CONVENTIONAL;
    unsigned int threshGensSinceImproved = 15;
    return pneatm::Population<std::complex<double>> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init_gen, resetValues_gen, activationFns_gen, inputsActivationFns_gen, outputsActivationFns_gen, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger, specific_genomes, dstType, speciationThreshInit, threshGensSinceImproved, stats_filepath);
}

pneatm::Population<std::complex<double>> SetupPopulation_dis (unsigned int popSize, unsigned int fft_output_sz, unsigned int max_recurrency, spdlog::logger* logger, const std::string& stats_filepath) {
    // nodes scheme setup
    std::vector<size_t> bias_sch (1, 2);
    std::vector<size_t> inputs_sch (1, fft_output_sz);
    std::vector<size_t> outputs_sch (1, 1);
    std::vector<std::vector<size_t>> hiddens_sch_init = {{1}};

    // bias values
    std::complex<double>* unit_real = new std::complex<double> (1.0, 0.0);
    std::complex<double>* unit_imag = new std::complex<double> (0.0, 1.0);
    bias_init_dis [0] = (void*) unit_real;
    bias_init_dis [1] = (void*) unit_imag;

    // reset values
    std::complex<double>* nulldouble = new std::complex<double> (0.0, 0.0);
    resetValues_dis [0] = (void*) nulldouble;

    // activation functions setup
    // bias & inputs
    inputsActivationFns_dis = std::vector<ActivationFnBase*> (bias_sch [0] + inputs_sch [0]);
    for (ActivationFnBase*& actfn : inputsActivationFns_dis) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &identity);
        actfn->setPrintingFunction (noprinting);
    }
    // output
    outputsActivationFns_dis = std::vector<ActivationFnBase*> (outputs_sch [0]);
    for (ActivationFnBase*& actfn : outputsActivationFns_dis) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &output_dis);
        actfn->setPrintingFunction (noprinting);
    }
    // hiddens
    activationFns_dis = std::vector<std::vector<std::vector<ActivationFnBase*>>> (1, std::vector<std::vector<ActivationFnBase*>> (1, std::vector<ActivationFnBase*> (5)));
    activationFns_dis [0][0][0] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][0]->setFunction ((void*) &identity);
    activationFns_dis [0][0][0]->setPrintingFunction (noprinting);
    activationFns_dis [0][0][0]->setMutationFunction (identity_mutation);
    activationFns_dis [0][0][1] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][1]->setFunction ((void*) &sigmoid_real);
    activationFns_dis [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns_dis [0][0][1]->setMutationFunction (sigmoid_mutation);
    activationFns_dis [0][0][2] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][2]->setFunction ((void*) &sigmoid_imag);
    activationFns_dis [0][0][2]->setPrintingFunction (sigmoid_printing);
    activationFns_dis [0][0][2]->setMutationFunction (sigmoid_mutation);
    activationFns_dis [0][0][3] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][3]->setFunction ((void*) &lrelu_real);
    activationFns_dis [0][0][3]->setPrintingFunction (lrelu_printing);
    activationFns_dis [0][0][3]->setMutationFunction (lrelu_mutation);
    activationFns_dis [0][0][4] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][4]->setFunction ((void*) &lrelu_imag);
    activationFns_dis [0][0][4]->setPrintingFunction (lrelu_printing);
    activationFns_dis [0][0][4]->setMutationFunction (lrelu_mutation);

    std::vector<genomeStruct_t> specific_genomes = {};

    unsigned int N_ConnInit = 2;
    double probRecuInit = 0.5;
    double weightExtremumInit = 10.0;
    unsigned int maxRecuInit = max_recurrency;
    double speciationThreshInit = 100.0;
    distanceFn dstType = CONVENTIONAL;
    unsigned int threshGensSinceImproved = 15;
    return pneatm::Population<std::complex<double>> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init_dis, resetValues_dis, activationFns_dis, inputsActivationFns_dis, outputsActivationFns_dis, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger, specific_genomes, dstType, speciationThreshInit, threshGensSinceImproved, stats_filepath);
}

pneatm::Population<std::complex<double>> LoadPopulation_gen (const std::string& filepath, unsigned int fft_output_sz, spdlog::logger* logger, const std::string& stats_filepath) {
    // nodes scheme setup
    std::vector<size_t> bias_sch (1, 2);
    std::vector<size_t> inputs_sch (1, fft_output_sz);
    std::vector<size_t> outputs_sch (1, fft_output_sz);

    // bias values
    std::complex<double>* unit_real = new std::complex<double> (1.0, 0.0);
    std::complex<double>* unit_imag = new std::complex<double> (0.0, 1.0);
    bias_init_gen [0] = (void*) unit_real;
    bias_init_gen [1] = (void*) unit_imag;

    // reset values
    std::complex<double>* nulldouble = new std::complex<double> (0.0, 0.0);
    resetValues_gen [0] = (void*) nulldouble;

    // activation functions setup
    // bias & inputs
    inputsActivationFns_gen = std::vector<ActivationFnBase*> (bias_sch [0] + inputs_sch [0]);
    for (ActivationFnBase*& actfn : inputsActivationFns_gen) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &identity);
        actfn->setPrintingFunction (noprinting);
    }
    // output
    outputsActivationFns_gen = std::vector<ActivationFnBase*> (outputs_sch [0]);
    for (ActivationFnBase*& actfn : outputsActivationFns_gen) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &identity);
        actfn->setPrintingFunction (noprinting);
    }
    // hiddens
    activationFns_gen = std::vector<std::vector<std::vector<ActivationFnBase*>>> (1, std::vector<std::vector<ActivationFnBase*>> (1, std::vector<ActivationFnBase*> (5)));
    activationFns_gen [0][0][0] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][0]->setFunction ((void*) &identity);
    activationFns_gen [0][0][0]->setPrintingFunction (noprinting);
    activationFns_gen [0][0][0]->setMutationFunction (identity_mutation);
    activationFns_gen [0][0][1] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][1]->setFunction ((void*) &sigmoid_real);
    activationFns_gen [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns_gen [0][0][1]->setMutationFunction (sigmoid_mutation);
    activationFns_gen [0][0][2] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][2]->setFunction ((void*) &sigmoid_imag);
    activationFns_gen [0][0][2]->setPrintingFunction (sigmoid_printing);
    activationFns_gen [0][0][2]->setMutationFunction (sigmoid_mutation);
    activationFns_gen [0][0][3] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][3]->setFunction ((void*) &lrelu_real);
    activationFns_gen [0][0][3]->setPrintingFunction (lrelu_printing);
    activationFns_gen [0][0][3]->setMutationFunction (lrelu_mutation);
    activationFns_gen [0][0][4] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_gen [0][0][4]->setFunction ((void*) &lrelu_imag);
    activationFns_gen [0][0][4]->setPrintingFunction (lrelu_printing);
    activationFns_gen [0][0][4]->setMutationFunction (lrelu_mutation);

    return pneatm::Population<std::complex<double>> (filepath, bias_init_gen, resetValues_gen, activationFns_gen, inputsActivationFns_gen, outputsActivationFns_gen, logger, stats_filepath);
}

pneatm::Population<std::complex<double>> LoadPopulation_dis (const std::string& filepath, unsigned int fft_output_sz, spdlog::logger* logger, const std::string& stats_filepath) {
    // nodes scheme setup
    std::vector<size_t> bias_sch (1, 2);
    std::vector<size_t> inputs_sch (1, fft_output_sz);
    std::vector<size_t> outputs_sch (1, 1);


    // bias values
    std::complex<double>* unit_real = new std::complex<double> (1.0, 0.0);
    std::complex<double>* unit_imag = new std::complex<double> (0.0, 1.0);
    bias_init_dis [0] = (void*) unit_real;
    bias_init_dis [1] = (void*) unit_imag;

    // reset values
    std::complex<double>* nulldouble = new std::complex<double> (0.0, 0.0);
    resetValues_dis [0] = (void*) nulldouble;

    // activation functions setup
    // bias & inputs
    inputsActivationFns_dis = std::vector<ActivationFnBase*> (bias_sch [0] + inputs_sch [0]);
    for (ActivationFnBase*& actfn : inputsActivationFns_dis) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &identity);
        actfn->setPrintingFunction (noprinting);
    }
    // output
    outputsActivationFns_dis = std::vector<ActivationFnBase*> (outputs_sch [0]);
    for (ActivationFnBase*& actfn : outputsActivationFns_dis) {
        actfn = new ActivationFn<std::complex<double>, std::complex<double>> ();
        actfn->setFunction ((void*) &output_dis);
        actfn->setPrintingFunction (noprinting);
    }
    // hiddens
    activationFns_dis = std::vector<std::vector<std::vector<ActivationFnBase*>>> (1, std::vector<std::vector<ActivationFnBase*>> (1, std::vector<ActivationFnBase*> (5)));
    activationFns_dis [0][0][0] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][0]->setFunction ((void*) &identity);
    activationFns_dis [0][0][0]->setPrintingFunction (noprinting);
    activationFns_dis [0][0][0]->setMutationFunction (identity_mutation);
    activationFns_dis [0][0][1] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][1]->setFunction ((void*) &sigmoid_real);
    activationFns_dis [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns_dis [0][0][1]->setMutationFunction (sigmoid_mutation);
    activationFns_dis [0][0][2] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][2]->setFunction ((void*) &sigmoid_imag);
    activationFns_dis [0][0][2]->setPrintingFunction (sigmoid_printing);
    activationFns_dis [0][0][2]->setMutationFunction (sigmoid_mutation);
    activationFns_dis [0][0][3] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][3]->setFunction ((void*) &lrelu_real);
    activationFns_dis [0][0][3]->setPrintingFunction (lrelu_printing);
    activationFns_dis [0][0][3]->setMutationFunction (lrelu_mutation);
    activationFns_dis [0][0][4] = new ActivationFn<std::complex<double>, std::complex<double>> ();
    activationFns_dis [0][0][4]->setFunction ((void*) &lrelu_imag);
    activationFns_dis [0][0][4]->setPrintingFunction (lrelu_printing);
    activationFns_dis [0][0][4]->setMutationFunction (lrelu_mutation);

    return pneatm::Population<std::complex<double>> (filepath, bias_init_dis, resetValues_dis, activationFns_dis, inputsActivationFns_dis, outputsActivationFns_dis, logger, stats_filepath);
}

void freePopulationsVars () {
    for (void*& x : bias_init_gen) {
        delete static_cast<std::complex<double>*> (x);
    }
    for (void*& x : resetValues_gen) {
        delete static_cast<std::complex<double>*> (x);
    }
    for (ActivationFnBase*& x : inputsActivationFns_gen) {
        delete x;
    }
    for (ActivationFnBase*& x : outputsActivationFns_gen) {
        delete x;
    }
    for (std::vector<std::vector<ActivationFnBase*>>& x : activationFns_gen) {
        for (std::vector<ActivationFnBase*>& y : x) {
            for (ActivationFnBase*& z : y) {
                delete z;
            }
        }
    }
    for (void*& x : bias_init_dis) {
        delete static_cast<std::complex<double>*> (x);
    }
    for (void*& x : resetValues_dis) {
        delete static_cast<std::complex<double>*> (x);
    }
    for (ActivationFnBase*& x : inputsActivationFns_dis) {
        delete x;
    }
    for (ActivationFnBase*& x : outputsActivationFns_dis) {
        delete x;
    }
    for (std::vector<std::vector<ActivationFnBase*>>& x : activationFns_dis) {
        for (std::vector<ActivationFnBase*>& y : x) {
            for (ActivationFnBase*& z : y) {
                delete z;
            }
        }
    }
}

std::function<pneatm::mutationParams_t (double)> SetupMutationParametersMaps (unsigned int max_recurrency) {
    pneatm::mutationParams_t explorationSet;
    explorationSet.nodes.rate = 0.3;
    explorationSet.nodes.monotypedRate = 1.0;
    explorationSet.nodes.monotyped.maxIterationsFindConnection = 100;
    //explorationSet.nodes.bityped.maxRecurrencyEntryConnection = ;
    //explorationSet.nodes.bityped.maxIterationsFindNode = ;
    explorationSet.activation_functions.rate = 0.15;
    explorationSet.connections.rate = 0.3;
    explorationSet.connections.reactivateRate = 0.4;
    explorationSet.connections.maxRecurrency = max_recurrency;
    explorationSet.connections.maxIterations = 100;
    explorationSet.connections.maxIterationsFindNode = 100;
    explorationSet.weights.rate = 0.1;
    explorationSet.weights.fullChangeRate = 0.3;
    explorationSet.weights.perturbationFactor = 1.0;
    return [=] (double fitness) {
        return explorationSet;
        UNUSED (fitness);
    };
}

void freeSpectrogram (std::vector<fftw_complex*> spectro) {
    for (fftw_complex*& fft : spectro) {
        fftw_free (fft);
    }
}

void freeSpectrograms (std::vector<std::vector<fftw_complex*>> spectros) {
    for (std::vector<fftw_complex*>& spectro : spectros) {
        freeSpectrogram (spectro);
    }
}

void saveAudioSignal (std::vector<std::vector<std::complex<double>>>& spectro, const char* filename, const unsigned int window_size, const unsigned int window_orerlap, const char* wisdom_filepath = "wisdom.fftw") {
    std::vector<std::vector<std::complex<double>>> spectro_copy = spectro;

    // Open the file
    SF_INFO fileInfo;
    fileInfo.samplerate = 44100;
    fileInfo.channels = 1;
    fileInfo.format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE;
    SNDFILE* file = sf_open (filename, SFM_WRITE, &fileInfo);
    if (!file) {
        std::cout << "Could not write file: " << filename << std::endl;
        return;
    }

    // Process the inverse FFT
    std::vector<double> audio;
    audio.reserve (spectro_copy.size () * (window_size - window_orerlap));

    fftw_complex* in = (fftw_complex*) fftw_malloc (sizeof (fftw_complex) * (int) (window_size / 2 + 1));
    double* out = new double [window_size];
    double* original_out = out;
    fftw_complex* fft_cast;
    int wisdom_exists;
    if (!(wisdom_exists = fftw_import_wisdom_from_filename (wisdom_filepath))) {
        std::cout << "No wisdom found for FFTW: one will be created which may takes some time." << std::endl;
    }
    fftw_plan plan = fftw_plan_dft_c2r_1d (window_size, in, out, FFTW_MEASURE);

    for (std::vector<std::complex<double>>& fft : spectro_copy) {

        fft_cast = reinterpret_cast<fftw_complex*> (fft.data ());

        // Fill in
        std::memcpy (in, fft_cast, sizeof (fftw_complex) * (int) (window_size / 2 + 1));

        // Execute FFT
        fftw_execute (plan);

        // Copy out to audio
        if (audio.size () > 0) {
            for (unsigned int offset = window_orerlap; offset > 0; offset--) {
                audio [audio.size () - offset] += *out++;
                audio [audio.size () - offset] /= 2.0;
            }
        } else {
            for (unsigned int offset = window_orerlap; offset > 0; offset--) {
                audio.push_back (*out++);
            }
        }
        for (unsigned int k = 0; k < window_size - window_orerlap; k++) {
            audio.push_back (*out++);
        }
        out = original_out;

    }

    if (!wisdom_exists) fftw_export_wisdom_to_filename (wisdom_filepath);
    delete[] out;
    fftw_free (in);
    fftw_destroy_plan (plan);

    // Limit the values to [-1.0, 1.0]
    const auto [min, max] = std::minmax_element (std::begin (audio), std::end (audio));
    if (*min < 0) *min -= 1;
    if (*max < 0) *max -= 1;
    const double ratio = std::max (*min, *max);
    for (double& value : audio) value /= ratio;

    // Save
    fileInfo.frames = audio.size ();
    if (sf_write_double (file, audio.data (), fileInfo.frames) != fileInfo.frames) {
        std::cout << "Not all the data has been saved to " << filename << std::endl;
    }
    sf_close (file);
}

std::vector<std::vector<std::vector<std::complex<double>>>> getSpectrograms (const std::string& root, const unsigned int window_size, const unsigned int window_orerlap, const char* wisdom_filepath = "wisdom.fftw") {

    std::vector<std::vector<std::vector<std::complex<double>>>> result;

    SF_INFO sfinfo;
    SNDFILE* sndfile = nullptr;

    double* in = new double [window_size];
    fftw_complex* out = (fftw_complex*) fftw_malloc (sizeof (fftw_complex) * (int) (window_size / 2 + 1));
    int wisdom_exists;
    if (!(wisdom_exists = fftw_import_wisdom_from_filename (wisdom_filepath))) {
        std::cout << "No wisdom found for FFTW: one will be created which may takes some time." << std::endl;
    }
    fftw_plan plan = fftw_plan_dft_r2c_1d (window_size, in, out, FFTW_MEASURE);

    std::filesystem::path directoryPath (root);
    if (std::filesystem::is_directory (directoryPath)) {
        for (const auto &filepath : std::filesystem::directory_iterator (directoryPath)) {
            if (filepath.is_regular_file ()) {

                sndfile = sf_open (filepath.path ().string ().c_str (), SFM_READ, &sfinfo);
                if (!sndfile) {
                    std::cerr << "Error opening file " << filepath << ": " << sf_strerror (nullptr) << std::endl;
                    continue;  // Move to the next file if an error occurs
                }

                result.push_back ({});

                double* audio = new double [sfinfo.frames];

                // Get a mono audio
                if (sfinfo.channels > 1) {

                    double* tempData = new double [sfinfo.frames * sfinfo.channels];
                    sf_readf_double(sndfile, tempData, sfinfo.frames);

                    for (unsigned int i = 0; i < sfinfo.frames; ++i) {
                        double monoSample = 0.0;
                        for (int j = 0; j < sfinfo.channels; ++j) {
                            monoSample += tempData [i * sfinfo.channels + j];
                        }
                        audio [i] = monoSample / sfinfo.channels;
                    }

                    delete[] tempData;
                } else {
                    sf_readf_double (sndfile, audio, sfinfo.frames);
                }
                sf_close(sndfile);

                for (unsigned int offset = 0; offset <= sfinfo.frames - window_size; offset += window_size - window_orerlap) {

                    // Fill in
                    std::memcpy (in, audio + offset, sizeof (double) * window_size);

                    // Execute FFT
                    fftw_execute (plan);

                    // Copy out to result
                    std::vector<std::complex<double>> out_copy ((int) (window_size / 2 + 1));
                    std::complex<double>* out_cast = reinterpret_cast<std::complex<double>*>(out);
                    std::copy(out_cast, out_cast + (int) (window_size / 2 + 1), out_copy.begin());
                    result.back ().push_back (out_copy);
                    result.back ().back ().shrink_to_fit ();

                }

                result.back ().shrink_to_fit ();
                delete[] audio;
            }
        }
    }

    result.shrink_to_fit ();

    if (!wisdom_exists) fftw_export_wisdom_to_filename (wisdom_filepath);

    delete[] in;
    fftw_free (out);
    fftw_destroy_plan (plan);

    return result;
}

std::vector<std::vector<std::complex<double>>> getSpectrogram (const std::vector<double>& audio, const unsigned int window_size, const unsigned int window_orerlap, const char* wisdom_filepath = "wisdom.fftw") {
    std::vector<std::vector<std::complex<double>>> result;

    double* in = new double [window_size];
    fftw_complex* out = (fftw_complex*) fftw_malloc (sizeof (fftw_complex) * (int) (window_size / 2 + 1));
    int wisdom_exists;
    if (!(wisdom_exists = fftw_import_wisdom_from_filename (wisdom_filepath))) {
        std::cout << "No wisdom found for FFTW: one will be created which may takes some time." << std::endl;
    }
    fftw_plan plan = fftw_plan_dft_r2c_1d (window_size, in, out, FFTW_MEASURE);


    for (unsigned int offset = 0; offset <= audio.size () - window_size; offset += window_size - window_orerlap) {

        // Fill in
        std::memcpy (in, audio.data () + offset, sizeof (double) * window_size);

        // Execute FFT
        fftw_execute (plan);

        // Copy out to result
        std::vector<std::complex<double>> out_copy ((int) (window_size / 2 + 1));
        std::complex<double>* out_cast = reinterpret_cast<std::complex<double>*>(out);
        std::copy(out_cast, out_cast + (int) (window_size / 2 + 1), out_copy.begin());
        result.push_back (out_copy);
        result.back ().shrink_to_fit ();

    }

    result.shrink_to_fit ();

    if (!wisdom_exists) fftw_export_wisdom_to_filename (wisdom_filepath);

    delete[] in;
    fftw_free (out);
    fftw_destroy_plan (plan);

    return result;
}

std::vector<std::vector<std::complex<double>>> getRandomSpectrogram (const std::string& root, const unsigned int window_size, const unsigned int window_orerlap, const char* wisdom_filepath = "wisdom.fftw") {

    // select a random file
    double n = 0.0;
    std::string entry;
    std::filesystem::path directoryPath (root);
    if (std::filesystem::is_directory (directoryPath)) {
        for (const auto &filepath : std::filesystem::directory_iterator (directoryPath)) {
            if (filepath.is_regular_file () && pneatm::Random_Double (0.0, 1.0, true, false) < 1.0 / n) {
                entry = filepath.path ().string ();
            }
            n += 1.0;
        }
    } else {
        std::cerr << root << " is not a path to a directory." << std::endl;
        throw 0;
    }

    // read it
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open (entry.c_str (), SFM_READ, &sfinfo);
    if (!sndfile) {
        std::cerr << "Error opening file " << entry << ": " << sf_strerror (nullptr) << std::endl;
        sf_close (sndfile);
        return getRandomSpectrogram (root, window_size, window_orerlap, wisdom_filepath);  // try another file. Warning: this may lead to too many calls!
    }

    std::vector<double> audio (sfinfo.frames);

    // Get a mono audio
    if (sfinfo.channels > 1) {

        double* tempData = new double [sfinfo.frames * sfinfo.channels];
        sf_readf_double(sndfile, tempData, sfinfo.frames);

        for (unsigned int i = 0; i < sfinfo.frames; ++i) {
            double monoSample = 0.0;
            for (int j = 0; j < sfinfo.channels; ++j) {
                monoSample += tempData [i * sfinfo.channels + j];
            }
            audio [i] = monoSample / sfinfo.channels;
        }

        delete[] tempData;
    } else {
        sf_readf_double (sndfile, audio.data (), sfinfo.frames);
    }
    sf_close(sndfile);

    return getSpectrogram (audio, window_size, window_orerlap, wisdom_filepath);
}

std::vector<double> White_Noise (const unsigned int lenght) {
    std::vector<double> output;
    output.reserve (lenght);
    for (unsigned int k = 0; k < lenght; k++) {
        output.push_back (pneatm::Random_Double (-1.0, 1.0));
    }
    return output;
}

void generate (Genome<std::complex<double>>& genome, std::vector<std::vector<std::complex<double>>> inputs, const char* filename, const unsigned int window_size, const unsigned int window_orerlap, const char* wisdom_filepath = "wisdom.fftw")  {
    std::vector<std::vector<std::complex<double>>> outputs;
    outputs.reserve (inputs.size ());
    for (const std::vector<std::complex<double>>& input : inputs) {
        genome.loadInputs (input);
        genome.runNetwork ();
        outputs.push_back (genome.template getOutputs<std::complex<double>> ());
    }

    if (!genome.isLocked ()) {
        saveAudioSignal (outputs, filename, window_size, window_orerlap, wisdom_filepath);
    } else {
        std::cout << "[ERROR] Genome has been locked during the generation process." << std::endl;
    }

    genome.resetMemory ();
}


#endif  // SETUP_HPP