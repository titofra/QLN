/**
    TODO:   add listening part before generation?
            opti: 1-declaration var
            opti: use of {}
*/

#include "setup.hpp"

int main (void) {
    // setup random seed
    srand ((int) time (0));

    const std::string rootPath = "dataset/data/piano/wav/";
    const unsigned int WINDOW_SIZE = 2048;  // 4.6ms
    const unsigned int WINDOW_OVERLAP = WINDOW_SIZE / 3;    // 33.3%
    const unsigned int TEST_N_SAMPLES = 5;
    const unsigned int NN_MEMORY_SIZE = 5 * 44100 / (WINDOW_SIZE - WINDOW_OVERLAP);    // 5s
    const unsigned int TRAIN_AUDIO_SIZE = NN_MEMORY_SIZE;    // 5s
    const unsigned int N_LOOP_GENERATE = 8;
    const unsigned int TRAIN_AUDIO_LEN = TRAIN_AUDIO_SIZE * (WINDOW_SIZE - WINDOW_OVERLAP); // do not change it
    const unsigned int FFT_OUTPUT_SIZE = WINDOW_SIZE / 2 + 1;   // do not change it
    const unsigned int POP_SIZE_GEN = 100;
    const unsigned int POP_SIZE_DIS = 100;
    const unsigned int MAX_THREADS_PNEATM = 4;

    // init pneatm logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    auto logger = spdlog::stdout_color_mt("console");
    //auto logger = spdlog::rotating_logger_mt("QLN_logger", "output/log.txt", 1048576 * 100, 500);

    // init populations
    pneatm::Population<std::complex<double>> generators = SetupPopulation_gen (POP_SIZE_GEN, FFT_OUTPUT_SIZE, NN_MEMORY_SIZE, logger.get (), "output/stats_gen.csv");
    pneatm::Population<std::complex<double>> discriminators = SetupPopulation_dis (POP_SIZE_DIS, FFT_OUTPUT_SIZE, NN_MEMORY_SIZE, logger.get (), "output/stats_dis.csv");
    /*pneatm::Population<std::complex<double>> generators = LoadPopulation_gen ("output/410_gen", FFT_OUTPUT_SIZE, logger.get (), "output/stats_gen.csv");
    pneatm::Population<std::complex<double>> discriminators = LoadPopulation_dis ("output/405_dis", FFT_OUTPUT_SIZE, logger.get (), "output/stats_dis.csv");*/

    // init mutation parameters
    std::function<pneatm::mutationParams_t (double)> paramsMap_gen = SetupMutationParametersMaps (NN_MEMORY_SIZE);
    std::function<pneatm::mutationParams_t (double)> paramsMap_dis = SetupMutationParametersMaps (NN_MEMORY_SIZE);

    bool generator_is_winning = true;
    std::unique_ptr<pneatm::Genome<std::complex<double>>> bestGenerator = generators.getGenome (-1).clone ();
    std::unique_ptr<pneatm::Genome<std::complex<double>>> bestDiscriminator = discriminators.getGenome (-1).clone ();


    /* INFERENCE */
    /*for (unsigned int k = 0; k < 3; k++) {
        generate (*bestGenerator, getSpectrogram (White_Noise (TRAIN_AUDIO_LEN), WINDOW_SIZE, WINDOW_OVERLAP), ("output_" + std::to_string (k) + ".wav").c_str (), WINDOW_SIZE, WINDOW_OVERLAP);
    }
    bestGenerator->print ();
    bestGenerator->draw ("/usr/share/fonts/opentype/SF/SF-Pro.ttf");
    return 0;*/


    while (generators.getGeneration () + discriminators.getGeneration () < 500) { // while goal is not reach
        logger->info ("generation {}", generators.getGeneration () + discriminators.getGeneration ());

        if (generator_is_winning) {


            std::vector<double> cumulated_losses_real (POP_SIZE_DIS, 0.0);
            std::vector<double> cumulated_losses_fake (POP_SIZE_DIS, 0.0);

            for (unsigned int k = 0; k < TEST_N_SAMPLES; k++) {


                /* TEST the discriminators on a GENERATED spectrogram */

                // The best generator generates a spectrogram from a white noise
                std::vector<std::vector<void*>> generated_spectrogram;
                std::vector<std::vector<void*>> generated_spectrogram_copy;
                generated_spectrogram.reserve (TRAIN_AUDIO_SIZE);
                const std::vector<std::vector<std::complex<double>>> gen_inputs = getSpectrogram (White_Noise (TRAIN_AUDIO_LEN), WINDOW_SIZE, WINDOW_OVERLAP);
                for (const std::vector<std::complex<double>>& inputs : gen_inputs) {
                    bestGenerator->loadInputs (inputs);
                    bestGenerator->runNetwork ();
                    generated_spectrogram.push_back (bestGenerator->getOutputs ());
                }

                bool spectro_is_generated = !bestGenerator->isLocked ();
                bestGenerator->resetMemory ();

                unsigned int Nb_loop = 1;
                while (Nb_loop < N_LOOP_GENERATE && spectro_is_generated) {
                    generated_spectrogram_copy = generated_spectrogram;
                    generated_spectrogram.clear ();
                    generated_spectrogram.reserve (TRAIN_AUDIO_SIZE);
                    for (const std::vector<void*>& inputs : generated_spectrogram_copy) {
                        bestGenerator->loadInputs (inputs);
                        bestGenerator->runNetwork ();
                        generated_spectrogram.push_back (bestGenerator->getOutputs ());
                    }

                    spectro_is_generated = !bestGenerator->isLocked ();
                    bestGenerator->resetMemory ();

                    Nb_loop++;
                }


                if (spectro_is_generated) {

                    // run the discriminators
                    discriminators.run (generated_spectrogram, nullptr, MAX_THREADS_PNEATM);

                    // get discriminators's output
                    for (std::pair<const unsigned int, std::unique_ptr<Genome<std::complex<double>>>>& discriminator : discriminators) {
                        // get the probability of the generated audio. Note that an ideal discriminator's output is 0 as it is not a real audio.
                        if (!discriminator.second->isLocked ()) {
                            cumulated_losses_fake [discriminator.first] += discriminator.second->template getOutput<std::complex<double>> (0).real ();
                        } else {
                            // the best discriminator raised a NaN, we give it the worst output
                            cumulated_losses_fake [discriminator.first] += 1.0;
                        }
                    }

                    // reset the discriminators's memory
                    discriminators.resetMemory ();

                } else {
                    // the best generator raised a NaN, we will give to each discriminators the maximum fitness, i.e. no loss
                    // add 0.0 for everyone in cumulated_losses_fake ...
                }


                /* TEST the discriminators on a REAL AUDIO */

                // setup discriminator's inputs
                const std::vector<std::vector<std::complex<double>>>& sample = getRandomSpectrogram (rootPath, WINDOW_SIZE, WINDOW_OVERLAP);
                const unsigned int offset = pneatm::Random_UInt (0, (unsigned int) sample.size () - TRAIN_AUDIO_SIZE - 1);
                std::vector<std::vector<std::complex<double>>> sample_cropped (sample.begin () + offset, sample.begin () + offset + TRAIN_AUDIO_SIZE);
                std::vector<std::vector<void*>> dis_inputs;
                dis_inputs.reserve (TRAIN_AUDIO_SIZE);
                for (std::vector<std::complex<double>>& inputs : sample_cropped) {
                    dis_inputs.push_back ({});
                    dis_inputs.back ().reserve (inputs.size ());
                    for (std::complex<double>& input : inputs) {
                        dis_inputs.back ().push_back (static_cast<void*> (&input));
                    }
                }

                // run the discriminators
                discriminators.run (dis_inputs, nullptr, MAX_THREADS_PNEATM);

                // get discriminators's output
                for (std::pair<const unsigned int, std::unique_ptr<Genome<std::complex<double>>>>& discriminator : discriminators) {
                    // get the probability of the real audio. Note that an ideal discriminator's output is 1 as it is a real audio.
                    if (!discriminator.second->isLocked ()) {
                        cumulated_losses_real [discriminator.first] += (1.0 - discriminator.second->template getOutput<std::complex<double>> (0).real ());
                    } else {
                        // the best generator raised a NaN, we give it the worst loss
                        cumulated_losses_real [discriminator.first] += 1.0;
                    }
                }

                // reset the discriminators's memory
                discriminators.resetMemory ();

            }


            /* GRADE the discriminators */
            for (std::pair<const unsigned int, std::unique_ptr<Genome<std::complex<double>>>>& discriminator : discriminators) {
                // grade the discriminator
                discriminator.second->setFitness (1.0 / (1e-10 + (cumulated_losses_real [discriminator.first] + cumulated_losses_fake [discriminator.first]) / (double) TEST_N_SAMPLES));  // 1 / (1e-10 + avg_loss) with avg_loss the average loss on real+fake (avg_loss is in [0.0, 2.0])
            }


            /* SPECIATE the discriminators */
            discriminators.speciate (5, 100, 0.3);


            /* SAVE */
            bestDiscriminator = discriminators.getGenome (-1).clone ();
            const double diss_loss_real = cumulated_losses_real [bestDiscriminator->getID ()] / TEST_N_SAMPLES;
            const double diss_loss_gen = cumulated_losses_fake [bestDiscriminator->getID ()] / TEST_N_SAMPLES;
            discriminators.save ("output/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_dis");


            /* LOG */
            logger->info ("dis_loss_real {}\tdis_loss_gen {}", diss_loss_real, diss_loss_gen);


            /* generate the NEW GENERATION of discriminators */
            discriminators.buildNextGen (paramsMap_dis, true, 0.5);


            /* CHECK if the generator is still better than the discriminators */
            if (diss_loss_real < 0.5 && diss_loss_gen < 0.5) {
                generator_is_winning = false;
            }


        } else {


            std::vector<double> cumulated_losses (POP_SIZE_GEN, 0.0);

            for (unsigned int k = 0; k < TEST_N_SAMPLES; k++) {

                /* generators GENERATE an audio */
            
                // input's setup
                std::vector<std::vector<std::complex<double>>> white_noise = getSpectrogram (White_Noise (TRAIN_AUDIO_LEN), WINDOW_SIZE, WINDOW_OVERLAP);
                std::vector<std::vector<void*>> inputs_com;
                inputs_com.reserve (TRAIN_AUDIO_SIZE);
                for (std::vector<std::complex<double>>& inputs : white_noise) {
                    inputs_com.push_back ({});
                    inputs_com.back ().reserve (FFT_OUTPUT_SIZE);
                    for (std::complex<double>& input : inputs) {
                        inputs_com.back ().push_back (static_cast<void*> (&input));
                    }
                }

                // get the generated spectrograms
                std::vector<std::vector<std::vector<void*>>> inputs_gen_void (POP_SIZE_GEN, inputs_com);
                std::vector<std::vector<std::vector<std::complex<double>>>> inputs_gen;
                std::vector<std::vector<std::vector<void*>>> outputs_gen;
                for (unsigned int loop = 0; loop < N_LOOP_GENERATE; loop++) {
                    // run the networks
                    generators.run (inputs_gen_void, &outputs_gen, MAX_THREADS_PNEATM, false);

                    // setup the new inputs
                    if (loop <= 0) {
                        white_noise.clear ();
                        inputs_com.clear ();
                    }
                    inputs_gen.clear ();
                    inputs_gen_void.clear ();
                    inputs_gen.reserve (POP_SIZE_GEN);
                    inputs_gen_void.reserve (POP_SIZE_GEN);
                    for (std::vector<std::vector<void*>>& x : outputs_gen) {
                        inputs_gen.push_back ({});
                        inputs_gen_void.push_back ({});
                        inputs_gen.back ().reserve (TRAIN_AUDIO_SIZE);
                        inputs_gen_void.back ().reserve (TRAIN_AUDIO_SIZE);
                        for (std::vector<void*>& y : x) {
                            inputs_gen.back ().push_back ({});
                            inputs_gen_void.back ().push_back ({});
                            inputs_gen.back ().back ().reserve (FFT_OUTPUT_SIZE);
                            inputs_gen_void.back ().back ().reserve (FFT_OUTPUT_SIZE);
                            for (void*& z : y) {
                                inputs_gen.back ().back ().push_back (*static_cast<std::complex<double>*> (z));
                                inputs_gen_void.back ().back ().push_back (static_cast<void*> (&inputs_gen.back ().back ().back ()));
                            }
                        }
                    }

                    // reset the memory
                    generators.resetMemory ();
                    outputs_gen.clear ();
                }

                for (std::pair<const unsigned int, std::unique_ptr<Genome<std::complex<double>>>>& generator : generators) {

                    if (inputs_gen_void [generator.first].size () > 0) {    // check if the genome has been locked (the former inputs_gen are now the final generated spectrograms)

                        // run the discriminator
                        for (std::vector<void*>& inputs : inputs_gen_void [generator.first]) {   // the former inputs_gen are now the final generated spectrograms
                            bestDiscriminator->loadInputs (inputs);
                            bestDiscriminator->runNetwork ();
                        }

                        if (!bestDiscriminator->isLocked ()) {
                            cumulated_losses [generator.first] += (1.0 - bestDiscriminator->template getOutput<std::complex<double>> (0).real ());
                        } else {
                            // discriminator diverged, we consider that the generator passed.
                            // add 0.0 to cumulated_losses [genome.first]
                        }

                        // reset the discriminator's memory
                        bestDiscriminator->resetMemory ();

                    } else {
                        // the generator raise a NaN value, it get the maximum loss
                        cumulated_losses [generator.first] += 1.0;
                    }

                }

            }


            /* GRADE the generators */
            for (std::pair<const unsigned int, std::unique_ptr<Genome<std::complex<double>>>>& generator : generators) {
                generator.second->setFitness (1.0 / (1e-10 + cumulated_losses [generator.first] / (double) TEST_N_SAMPLES));  // 1 / (1e-10 + avg_loss) with avg_loss the average loss (avg_loss is in [0.0, 1.0])
            }


            /* SPECIATE the generators */
            generators.speciate (5, 100, 0.3);


            /* SAVE */
            bestGenerator = generators.getGenome (-1).clone ();
            const double gen_loss = cumulated_losses [bestGenerator->getID ()] / TEST_N_SAMPLES;
            generators.save ("output/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_gen");


            /* LOG */
            logger->info ("gen_loss {}", gen_loss);


            /* generate the NEW GENERATION of generators */
            generators.buildNextGen (paramsMap_gen, true, 0.5);


            /* CHECK if the discriminator is still better than the generators */
            if (gen_loss < 0.5) {
                generator_is_winning = true;
            }


        }
    }

    freePopulationsVars ();

    return 0;
}