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


/* DEFINITIONS */

#define UNUSED(expr) do { (void) (expr); } while (0)

// parameters
typedef struct activationFnParams {
    double alpha = Random_Double (-10.0, 10.0);
    double beta = Random_Double (-10.0, 10.0);
} activationFnParams_t;

// activation functions
std::function<double (double, activationFnParams_t*)> identity = [] (double x, activationFnParams_t* params) -> double {
    return x;
    UNUSED (params);
};
std::function<double (double, activationFnParams_t*)> output_sigmoid_gen = [] (double x, activationFnParams_t* params) -> double {
    return 2.0 / (1.0 + std::exp(-1.0 * x)) - 1.0;
    UNUSED (params);
};
std::function<double (double, activationFnParams_t*)> output_sigmoid_dis = [] (double x, activationFnParams_t* params) -> double {
    return 1.0 / (1.0 + std::exp(-1.0 * x));
    UNUSED (params);
};
std::function<double (double, activationFnParams_t*)> sigmoid = [] (double x, activationFnParams_t* params) -> double {
    return 1.0 / (1.0 + std::exp(params->alpha * (x - params->beta)));
};

// mutation functions
std::function<void (activationFnParams_t*, double)> identity_mutation = [] (activationFnParams_t* params, double fitness) -> void {
    UNUSED (params);
    UNUSED (fitness);
};
std::function<void (activationFnParams_t*, double)> sigmoid_mutation = [] (activationFnParams_t* params, double fitness) -> void {
    if (Random_Double (0.0, 1.0, true, false) < 0.3) {
        // reset values
        params->alpha = Random_Double (-10.0, 10.0);
        params->beta = Random_Double (-10.0, 10.0);
    } else {
        // perturb values
        params->alpha += params->alpha * Random_Double (-1.0, 1.0);
        params->beta += params->beta * Random_Double (-1.0, 1.0);
    }
    UNUSED (fitness);
};

// printing functions
std::function<void (activationFnParams_t*, std::string)> noprinting = [] (activationFnParams_t* params, std::string prefix) -> void {
    UNUSED (params);
    UNUSED (prefix);
};
std::function<void (activationFnParams_t*, std::string)> sigmoid_printing = [] (activationFnParams_t* params, std::string prefix) -> void {
    std::cout << prefix << "alpha = " << params->alpha << "   beta = " << params->beta;
};


/* SETUP FUNCTIONS */

pneatm::Population<double> SetupPopulation_gen (unsigned int popSize, spdlog::logger* logger, const std::string& stats_filename) {
    // nodes scheme setup
    std::vector<size_t> bias_sch = {1};
    std::vector<size_t> inputs_sch = {1};
    std::vector<size_t> outputs_sch = {1};
    std::vector<std::vector<size_t>> hiddens_sch_init = {{3}};

    // bias values
    std::vector<void*> bias_init;
    double* unitdouble = new double (1.0);
    bias_init.push_back ((void*) unitdouble);

    // reset values
    std::vector<void*> resetValues;
    double* nulldouble = new double (0.0);
    resetValues.push_back ((void*) nulldouble);

    // activation functions setup
    // bias & inputs
    std::vector<ActivationFnBase*> inputsActivationFns;
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // output
    std::vector<ActivationFnBase*> outputsActivationFns;
    outputsActivationFns.push_back (new ActivationFn<double, double> ());
    outputsActivationFns.back ()->setFunction ((void*) &output_sigmoid_gen);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // hiddens
    std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0][0]->setFunction ((void*) &identity);
    activationFns [0][0][1]->setFunction ((void*) &sigmoid);
    activationFns [0][0][0]->setPrintingFunction (noprinting);
    activationFns [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns [0][0][0]->setMutationFunction (identity_mutation);
    activationFns [0][0][1]->setMutationFunction (sigmoid_mutation);

    unsigned int N_ConnInit = 4;
    double probRecuInit = 0.0;
    double weightExtremumInit = 2.0;
    unsigned int maxRecuInit = 0;
    double speciationThreshInit = 20.0;
    distanceFn dstType = CONVENTIONAL;
    unsigned int threshGensSinceImproved = 15;
    return pneatm::Population<double> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, inputsActivationFns, outputsActivationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger, dstType, speciationThreshInit, threshGensSinceImproved, stats_filename);
}

pneatm::Population<double> SetupPopulation_dis (unsigned int popSize, spdlog::logger* logger, const std::string& stats_filename) {
    // nodes scheme setup
    std::vector<size_t> bias_sch = {1};
    std::vector<size_t> inputs_sch = {1};
    std::vector<size_t> outputs_sch = {1};
    std::vector<std::vector<size_t>> hiddens_sch_init = {{3}};

    // bias values
    std::vector<void*> bias_init;
    double* unitdouble = new double (1.0);
    bias_init.push_back ((void*) unitdouble);

    // reset values
    std::vector<void*> resetValues;
    double* nulldouble = new double (0.0);
    resetValues.push_back ((void*) nulldouble);

    // activation functions setup
    // bias & inputs
    std::vector<ActivationFnBase*> inputsActivationFns;
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // output
    std::vector<ActivationFnBase*> outputsActivationFns;
    outputsActivationFns.push_back (new ActivationFn<double, double> ());
    outputsActivationFns.back ()->setFunction ((void*) &output_sigmoid_dis);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // hiddens
    std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0][0]->setFunction ((void*) &identity);
    activationFns [0][0][1]->setFunction ((void*) &sigmoid);
    activationFns [0][0][0]->setPrintingFunction (noprinting);
    activationFns [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns [0][0][0]->setMutationFunction (identity_mutation);
    activationFns [0][0][1]->setMutationFunction (sigmoid_mutation);

    unsigned int N_ConnInit = 4;
    double probRecuInit = 0.0;
    double weightExtremumInit = 2.0;
    unsigned int maxRecuInit = 0;
    double speciationThreshInit = 20.0;
    distanceFn dstType = CONVENTIONAL;
    unsigned int threshGensSinceImproved = 15;
    return pneatm::Population<double> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, inputsActivationFns, outputsActivationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger, dstType, speciationThreshInit, threshGensSinceImproved, stats_filename);
}

pneatm::Population<double> LoadPopulation_gen (const std::string& filename, spdlog::logger* logger, const std::string& stats_filename) {
    // bias values
    std::vector<void*> bias_init;
    double* unitdouble = new double (1.0);
    bias_init.push_back ((void*) unitdouble);

    // reset values
    std::vector<void*> resetValues;
    double* nulldouble = new double (0.0);
    resetValues.push_back ((void*) nulldouble);

    // activation functions setup
    // bias & inputs
    std::vector<ActivationFnBase*> inputsActivationFns;
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // output
    std::vector<ActivationFnBase*> outputsActivationFns;
    outputsActivationFns.push_back (new ActivationFn<double, double> ());
    outputsActivationFns.back ()->setFunction ((void*) &output_sigmoid_gen);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // hiddens
    std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0][0]->setFunction ((void*) &identity);
    activationFns [0][0][1]->setFunction ((void*) &sigmoid);
    activationFns [0][0][0]->setPrintingFunction (noprinting);
    activationFns [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns [0][0][0]->setMutationFunction (identity_mutation);
    activationFns [0][0][1]->setMutationFunction (sigmoid_mutation);

    return pneatm::Population<double> (filename, bias_init, resetValues, activationFns, inputsActivationFns, outputsActivationFns, logger, stats_filename);
}

pneatm::Population<double> LoadPopulation_dis (const std::string& filename, spdlog::logger* logger, const std::string& stats_filename) {
    // bias values
    std::vector<void*> bias_init;
    double* unitdouble = new double (1.0);
    bias_init.push_back ((void*) unitdouble);

    // reset values
    std::vector<void*> resetValues;
    double* nulldouble = new double (0.0);
    resetValues.push_back ((void*) nulldouble);

    // activation functions setup
    // bias & inputs
    std::vector<ActivationFnBase*> inputsActivationFns;
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    inputsActivationFns.push_back (new ActivationFn<double, double> ());
    inputsActivationFns.back ()->setFunction ((void*) &identity);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // output
    std::vector<ActivationFnBase*> outputsActivationFns;
    outputsActivationFns.push_back (new ActivationFn<double, double> ());
    outputsActivationFns.back ()->setFunction ((void*) &output_sigmoid_dis);
    inputsActivationFns.back ()->setPrintingFunction (noprinting);
    // hiddens
    std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0].push_back (new ActivationFn<double, double> ());
    activationFns [0][0][0]->setFunction ((void*) &identity);
    activationFns [0][0][1]->setFunction ((void*) &sigmoid);
    activationFns [0][0][0]->setPrintingFunction (noprinting);
    activationFns [0][0][1]->setPrintingFunction (sigmoid_printing);
    activationFns [0][0][0]->setMutationFunction (identity_mutation);
    activationFns [0][0][1]->setMutationFunction (sigmoid_mutation);

    return pneatm::Population<double> (filename, bias_init, resetValues, activationFns, inputsActivationFns, outputsActivationFns, logger, stats_filename);
}

std::function<pneatm::mutationParams_t (double)> SetupMutationParametersMaps (unsigned int AUDIO_LEN) {
    pneatm::mutationParams_t explorationSet;
    explorationSet.nodes.rate = 0.2;
    explorationSet.nodes.monotypedRate = 1.0;
    explorationSet.nodes.monotyped.maxIterationsFindConnection = 100;
    //explorationSet.nodes.bityped.maxRecurrencyEntryConnection = ;
    //explorationSet.nodes.bityped.maxIterationsFindNode = ;
    explorationSet.activation_functions.rate = 0.1;
    explorationSet.connections.rate = 0.2;
    explorationSet.connections.reactivateRate = 0.6;
    explorationSet.connections.maxRecurrency = AUDIO_LEN;
    explorationSet.connections.maxIterations = 100;
    explorationSet.connections.maxIterationsFindNode = 100;
    explorationSet.weights.rate = 0.1;
    explorationSet.weights.fullChangeRate = 0.2;
    explorationSet.weights.perturbationFactor = 1.0;
    return [=] (double fitness) {
        return explorationSet;
        UNUSED (fitness);
    };
}

std::vector<double> getAudioSignal (const char* filename) {
    SF_INFO fileInfo;
    SNDFILE* file = sf_open (filename, SFM_READ, &fileInfo);
    if (!file) {
        return {};
    }

    std::vector<double> samples (fileInfo.channels * fileInfo.frames);
    if (sf_readf_double (file, samples.data (), fileInfo.frames) != fileInfo.frames) {
        sf_close (file);
        return {};
    }

    std::vector<double> audio_signal (fileInfo.frames);
    if (fileInfo.channels > 1) {
        // == 2 stereo
        for (sf_count_t i = 0; i < fileInfo.frames; ++i) {
            audio_signal [i] = (samples [2 * i] + samples [2 * i + 1]) / 2.0f;
        }
    } else {
        // == 1 mono
        audio_signal = samples;
    }

    sf_close (file);
    return audio_signal;
}

std::vector<std::vector<double>> loadSamples (const std::string& root) {
    std::vector<std::vector<double>> result;

    std::filesystem::path directoryPath (root);

    if (std::filesystem::is_directory(directoryPath)) {
        for (const auto &entry : std::filesystem::directory_iterator(directoryPath)) {
            if (std::filesystem::is_regular_file(entry.status())) {
                std::vector<double> sample = getAudioSignal ((root + entry.path().filename().string()).c_str());

                result.push_back (sample);

            }
        }
    }
    
    return result;
}

std::vector<double> getRandomSample (const std::string& root) {
    std::vector<std::string> filePaths;
    std::filesystem::path directoryPath (root);
    if (std::filesystem::is_directory (directoryPath)) {
        for (const auto &entry : std::filesystem::directory_iterator (directoryPath)) {
            if (entry.is_regular_file ()) {
                filePaths.push_back (entry.path ().string ());
            }
        }
    }

    std::vector<double> sample = getAudioSignal ((filePaths [Random_UInt (0, (unsigned int) filePaths.size () - 1)]).c_str());

    return sample;
}

std::vector<std::vector<double>> getSamples (const std::string& root) {
    std::vector<std::vector<double>> samples;
    std::filesystem::path directoryPath (root);
    if (std::filesystem::is_directory (directoryPath)) {
        for (const auto &entry : std::filesystem::directory_iterator (directoryPath)) {
            if (entry.is_regular_file ()) {
                samples.push_back (getAudioSignal (entry.path ().string ().c_str()));
            }
        }
    }

    return samples;
}

void saveAudioSignal (const std::vector<double>& audio, const char* filename) {
    SF_INFO fileInfo;
    fileInfo.samplerate = 44100;
    fileInfo.channels = 1;
    fileInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* file = sf_open (filename, SFM_WRITE, &fileInfo);
    if (!file) {
        std::cout << "Could not write file: " << filename << std::endl;
        return;
    }

    fileInfo.frames = audio.size ();

    if (sf_write_double (file, audio.data (), fileInfo.frames) != fileInfo.frames) {
        std::cout << "Not all the data has been saved to " << filename << std::endl;
    }

    sf_close (file);
}

std::vector<double> white_noise (const unsigned int lenght) {
    std::vector<double> output;
    output.reserve (lenght);
    for (unsigned int k = 0; k < lenght; k++) {
        output.push_back (pneatm::Random_Double (-1.0, 1.0));
    }
    return output;
}

void generate (Genome<double>& genome, std::vector<double> input, const char* filename)  {
    std::vector<double> output;
    output.reserve (input.size ());
    for (double value : input) {
        genome.loadInput (value, 0);
        genome.runNetwork ();
        output.push_back (genome.template getOutput<double> (0));
    }

    genome.resetMemory ();

    if (!genome.isLocked ()) {
        saveAudioSignal (output, filename);
    }

}


#endif  // SETUP_HPP