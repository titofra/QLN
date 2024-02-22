#include "setup.hpp"

int main (void) {
    srand ((int) time (0));	// init seed for rand function

    // constants
    const unsigned int AUDIO_LEN = 44100 * 2;
    const unsigned int TEST_N_SAMPLES = 5;
    const unsigned int MAX_THREADS = 0; // default
    const unsigned int popSize_gen = 100;
    const unsigned int popSize_dis = 100;
    const std::string rootPath = "dataset/data/piano/wav/";

    // init pneatm logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    auto logger = spdlog::stdout_color_mt("console");
    //auto logger = spdlog::rotating_logger_mt("AGPNet_logger", "logs/log.txt", 1048576 * 100, 500);

    // init populations
    pneatm::Population<double> generators = SetupPopulation_gen (popSize_gen, logger.get (), "save/stats_gen.csv");
    pneatm::Population<double> discriminators = SetupPopulation_dis (popSize_dis, logger.get (), "save/stats_dis.csv");
    /*pneatm::Population<double> generators = LoadPopulation_gen ("save/499_gen", logger.get (), "save/stats_gen.csv");
    pneatm::Population<double> discriminators = LoadPopulation_dis ("save/498_dis", logger.get (), "save/stats_dis.csv");*/

    // init mutation parameters
    std::function<pneatm::mutationParams_t (double)> paramsMap_gen = SetupMutationParametersMaps (AUDIO_LEN);
    std::function<pneatm::mutationParams_t (double)> paramsMap_dis = SetupMutationParametersMaps (AUDIO_LEN);

    bool generator_is_winning = true;
    std::unique_ptr<pneatm::Genome<double>> bestGenerator = generators.getGenome (-1).clone ();
    std::unique_ptr<pneatm::Genome<double>> bestDiscriminator = discriminators.getGenome (-1).clone ();



    /* INFERENCE */
    /*generate (*bestGenerator, white_noise (AUDIO_LEN), "output.wav");
    bestGenerator->print ();
    bestGenerator->draw ("/usr/share/fonts/opentype/SF/SF-Pro.ttf");
    return 0;*/



    const std::vector<std::vector<double>> samples = getSamples (rootPath);

    while (generators.getGeneration () + discriminators.getGeneration () < 500) { // while goal is not reach
        logger->info ("generation {}", generators.getGeneration () + discriminators.getGeneration ());

        if (generator_is_winning) {

            std::vector<double> cumulated_losses_real (popSize_dis, 0.0);
            std::vector<double> cumulated_losses_fake (popSize_dis, 0.0);

            for (unsigned int k = 0; k < TEST_N_SAMPLES; k++) {


                /* TEST the discriminators on a GENERATED AUDIO */

                // The best generator generates an audio from a white noise
                std::vector<double> generated_audio;
                generated_audio.reserve (AUDIO_LEN);
                for (unsigned int k = 0; k < AUDIO_LEN; k++) {
                    bestGenerator->loadInput (pneatm::Random_Double (-1.0, 1.0), 0);
                    bestGenerator->runNetwork ();
                    generated_audio.push_back (bestGenerator->template getOutput<double> (0));
                }

                const bool audio_is_generated = !bestGenerator->isLocked ();

                // reset genome's memory
                bestGenerator->resetMemory ();

                std::vector<std::vector<void*>> dis_inputs;
                dis_inputs.reserve (AUDIO_LEN);
                if (audio_is_generated) {
                    // setup discriminator's inputs
                    for (double& input : generated_audio) {
                        dis_inputs.push_back (std::vector<void*> (1, static_cast<void*> (&input)));
                    }

                    // run the discriminators
                    discriminators.run (dis_inputs, nullptr, MAX_THREADS);

                    // get discriminators's output
                    for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& discriminator : discriminators) {
                        // get the probability of the generated audio. Note that an ideal discriminator's output is 0 as it is not a real audio.
                        if (!discriminator.second->isLocked ()) {
                            cumulated_losses_fake [discriminator.first] += discriminator.second->template getOutput<double> (0);
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
                const std::vector<double>& sample = samples [pneatm::Random_UInt (0, (unsigned int) samples.size () - 1)];
                const unsigned int start = pneatm::Random_UInt (0, (unsigned int) sample.size () - AUDIO_LEN - 1);
                std::vector<double> real_audio (sample.begin () + start, sample.begin () + start + AUDIO_LEN);
                dis_inputs.clear ();
                for (double& input : real_audio) {
                    dis_inputs.push_back (std::vector<void*> (1, static_cast<void*> (&input)));
                }

                // run the discriminators
                discriminators.run (dis_inputs, nullptr, MAX_THREADS);

                // get discriminators's output
                for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& discriminator : discriminators) {
                    // get the probability of the real audio. Note that an ideal discriminator's output is 1 as it is a real audio.
                    if (!discriminator.second->isLocked ()) {
                        cumulated_losses_real [discriminator.first] += (1.0 - discriminator.second->template getOutput<double> (0));
                    } else {
                        // the best generator raised a NaN, we give it the worst loss
                        cumulated_losses_real [discriminator.first] += 1.0;
                    }
                }

                // reset the discriminators's memory
                discriminators.resetMemory ();

            }


            /* GRADE the discriminators */
            for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& discriminator : discriminators) {
                // grade the discriminator
                discriminator.second->setFitness (1.0 / (1e-10 + (cumulated_losses_real [discriminator.first] + cumulated_losses_fake [discriminator.first]) / (double) TEST_N_SAMPLES));  // 1 / (1e-10 + avg_loss) with avg_loss the average loss on real+fake (avg_loss is in [0.0, 2.0])
            }


            /* SPECIATE the discriminators */
            discriminators.speciate (5, 100, 0.3);


            /* SAVE */
            bestDiscriminator = discriminators.getGenome (-1).clone ();
            const double diss_loss_real = cumulated_losses_real [bestDiscriminator->getID ()] / TEST_N_SAMPLES;
            const double diss_loss_gen = cumulated_losses_fake [bestDiscriminator->getID ()] / TEST_N_SAMPLES;
            discriminators.save ("save/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_dis");


            /* LOG */
            logger->info ("dis_loss_real {} \t dis_loss_gen {}", diss_loss_real, diss_loss_gen);


            /* generate the NEW GENERATION of discriminators */
            discriminators.buildNextGen (paramsMap_dis, true, 0.7);


            /* CHECK if the generator is still better than the discriminators */
            if (diss_loss_real < 0.5 && diss_loss_gen < 0.5) {
                generator_is_winning = false;
            }



        } else {

            std::vector<double> cumulated_losses (popSize_gen, 0.0);

            for (unsigned int k = 0; k < TEST_N_SAMPLES; k++) {

                /* generators GENERATE an audio */
            
                // white noise input's setup
                std::vector<double> white_noise;
                white_noise.reserve (AUDIO_LEN);
                std::vector<std::vector<void*>> inputs;
                inputs.reserve (AUDIO_LEN);
                for (unsigned int i = 0 ; i < AUDIO_LEN; i++) {
                    white_noise.push_back (pneatm::Random_Double (-1.0, 1.0));
                    inputs.push_back (std::vector<void*> {static_cast<void*> (&white_noise.back ())});
                }

                // run the networks
                std::vector<std::vector<void*>> outputs;
                generators.run (inputs, &outputs, MAX_THREADS);

                // get the resulting audio
                std::vector<std::vector<double>> audio;
                audio.reserve (popSize_gen);
                for (std::vector<void*>& output : outputs) {
                    if (output.size () > 0) {
                        audio.push_back (*static_cast<std::vector<double>*> (output [0]));
                    } else {
                        // the genome has been locked
                        audio.push_back ({});
                    }
                }

                // keep in memory the locked genomes
                std::vector<bool> locked_generators (popSize_gen, false);
                for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& generator : generators) {
                    locked_generators [generator.first] = generator.second->isLocked ();
                }

                // reset genomes's memory
                generators.resetMemory ();

                for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& generator : generators) {

                    if (!locked_generators [generator.first]) {

                        // run the discriminator
                        for (double input : audio [generator.first]) {
                            bestDiscriminator->loadInput (input, 0);
                            bestDiscriminator->runNetwork ();
                        }

                        if (!bestDiscriminator->isLocked ()) {
                            cumulated_losses [generator.first] += (1.0 - bestDiscriminator->template getOutput<double> (0));
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
            for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& generator : generators) {
                generator.second->setFitness (1.0 / (1e-10 + cumulated_losses [generator.first] / (double) TEST_N_SAMPLES));  // 1 / (1e-10 + avg_loss) with avg_loss the average loss (avg_loss is in [0.0, 1.0])
            }


            /* SPECIATE the generators */
            generators.speciate (5, 100, 0.3);


            /* SAVE */
            bestGenerator = generators.getGenome (-1).clone ();
            const double gen_loss = cumulated_losses [bestGenerator->getID ()] / TEST_N_SAMPLES;
            generators.save ("save/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_gen");


            /* LOG */
            logger->info ("gen_loss {}", gen_loss);


            /* generate the NEW GENERATION of generators */
            generators.buildNextGen (paramsMap_gen, true, 0.7);


            /* CHECK if the discriminator is still better than the generators */
            if (gen_loss < 0.5) {
                generator_is_winning = true;
            }


        }


        /* INFERENCE */
        if ((generators.getGeneration () + discriminators.getGeneration ()) % 100 == 0) generate (*bestGenerator, white_noise (AUDIO_LEN), (std::to_string(generators.getGeneration () + discriminators.getGeneration ()) + ".wav").c_str ());
    }

    return 0;
}