#include "setup.hpp"

int main (void) {
    srand ((int) time (0));	// init seed for rand function

    // constants
    const unsigned int N_CALLS_LISTEN = 44100 * 2;
    const unsigned int N_CALLS_GUESS = 18000;
    const unsigned int MAX_THREADS = 0; // default
    const unsigned int popSize_gen = 100;
    const unsigned int popSize_dis = 100;
    const std::string rootPath = "dataset/data/wav/";

    // init pneatm logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    auto logger = spdlog::stdout_color_mt("console");
    //auto logger = spdlog::rotating_logger_mt("AGPNet_logger", "logs/log.txt", 1048576 * 100, 500);

    // init populations
    pneatm::Population<double> generators = SetupPopulation (popSize_gen, logger.get (), "save/stats_gen.csv");
    pneatm::Population<double> discriminators = SetupPopulation (popSize_dis, logger.get (), "save/stats_dis.csv");
    /*pneatm::Population<double> generators = LoadPopulation ("save/0_gen", logger.get (), "save/stats_gen.csv");
    pneatm::Population<double> discriminators = LoadPopulation ("save/2_dis", logger.get (), "save/stats_dis.csv");*/

    // init mutation parameters
    std::function<pneatm::mutationParams_t (double)> paramsMap_gen = SetupMutationParametersMaps (N_CALLS_LISTEN);
    std::function<pneatm::mutationParams_t (double)> paramsMap_dis = SetupMutationParametersMaps (N_CALLS_LISTEN);

    bool generator_is_winning = true;
    std::unique_ptr<pneatm::Genome<double>> bestGenerator = generators.getGenome (-1).clone ();
    std::unique_ptr<pneatm::Genome<double>> bestDiscriminator = discriminators.getGenome (-1).clone ();

    while (generators.getGeneration () + discriminators.getGeneration () < 9500) { // while goal is not reach
        logger->info ("generation {}", generators.getGeneration () + discriminators.getGeneration ());

        // Prepare the samples
        std::vector<double> sample = getRandomSample (rootPath);
        std::vector<double> sample_listen (sample.begin (), sample.begin () + N_CALLS_LISTEN);
        std::vector<double> sample_target (sample.begin (), sample.begin () + N_CALLS_LISTEN + N_CALLS_GUESS);

        if (generator_is_winning) {


            /* TEST the discriminators on a GENERATED AUDIO */

            // Run the networks over the listening part e.g. it don't predict anything, it just listen for the audio.
            for (const double& input : sample_listen) {
                bestGenerator->loadInput (input, 0);
                bestGenerator->runNetwork ();
			}

            // Run the network over the predicting part e.g. it predict the audio.
            std::vector<double> generated_audio;
            for (unsigned int k = 0; k < N_CALLS_GUESS; k++) {
                generated_audio.push_back (bestGenerator->template getOutput<double> (0));
                bestGenerator->loadInput (generated_audio.back (), 0);
			}

            const bool audio_is_generated = !bestGenerator->isLocked ();

            // reset genome's memory
            bestGenerator->resetMemory ();

            std::vector<std::vector<void*>> dis_inputs;
            std::vector<double> dis_guess_on_generated;
            if (audio_is_generated) {
                // get the resulting audio
                std::vector<double> audio = sample_listen;
                audio.insert (audio.end (), generated_audio.begin (), generated_audio.end ());

                // setup discriminator's inputs
                for (double& input : audio) {
                    dis_inputs.push_back (std::vector<void*> (1, static_cast<void*> (&input)));
                }

                // run the discriminators
                discriminators.run (dis_inputs, nullptr, MAX_THREADS);

                // get discriminators's output
                for (unsigned int id = 0; id < popSize_dis; id++) {
                    // get the probability of the generated audio. Note that an ideal discriminator's output is 0 as it is not a real audio.
                    Genome<double>* discriminator = discriminators.getpGenome (id);
                    if (!discriminator->isLocked ()) {
                        dis_guess_on_generated.push_back (discriminator->template getOutput<double> (0));
                    } else {
                        // the best generator raised a NaN, we give it the worst output
                        dis_guess_on_generated.push_back (1.0);
                    }
                }

                // reset the discriminators's memory
                discriminators.resetMemory ();

            } else {
                // the best generator raised a NaN, we will give to each discriminators the maximum fitness
                for (unsigned int id = 0; id < popSize_dis; id++) {
                    // get the probability of the generated audio. Note that an ideal discriminator's output is 0 as it is not a real audio.
                    dis_guess_on_generated.push_back (0.0);
                }
            }


            /* TEST the discriminators on a REAL AUDIO */

            // setup discriminator's inputs
            dis_inputs.clear ();
            for (double& input : sample_target) {
                dis_inputs.push_back (std::vector<void*> (1, static_cast<void*> (&input)));
            }

            // run the discriminators
            discriminators.run (dis_inputs, nullptr, MAX_THREADS);

            // get discriminators's output
            std::vector<double> dis_guess_on_real;
            for (unsigned int id = 0; id < popSize_dis; id++) {
                // get the probability of the real audio. Note that an ideal discriminator's output is 1 as it is a real audio.
                Genome<double>* discriminator = discriminators.getpGenome (id);
                if (!discriminator->isLocked ()) {
                    dis_guess_on_real.push_back (discriminator->template getOutput<double> (0));
                } else {
                    // the best generator raised a NaN, we give it the worst output
                    dis_guess_on_real.push_back (0.0);
                }
            }

            // reset the discriminators's memory
            discriminators.resetMemory ();


            /* GRADE the discriminators */
            for (unsigned int id = 0; id < popSize_dis; id++) {
                const double score =
                    (1.0 - dis_guess_on_real [id])
                    + dis_guess_on_generated [id];
                // grade the discriminator
                if (score > 0.0) {
                    discriminators.getGenome (id).setFitness (1.0 / score);
                } else {
                    discriminators.getGenome (id).setFitness (DBL_MAX);
                }
            }


            /* SPECIATE the discriminators */
            discriminators.speciate (5, 100, 0.3);


            /* SAVE */
            bestDiscriminator = discriminators.getGenome (-1).clone ();
            discriminators.save ("save/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_dis");


            /* generate the NEW GENERATION of discriminators */
            discriminators.crossover (true, 0.7);


            /* MUTATE the discriminators */
            discriminators.mutate (paramsMap_dis);


            /* CHECK if the generator is still better than the discriminators */
            if (dis_guess_on_real [bestDiscriminator->getID ()] > 0.5 && dis_guess_on_generated [bestDiscriminator->getID ()] < 0.5) {
                generator_is_winning = false;
            }

            /* LOG */
            logger->info ("dis_loss_real {} \t dis_loss_gen {}", (1.0 - dis_guess_on_real [bestDiscriminator->getID ()]), dis_guess_on_generated [bestDiscriminator->getID ()]);


        } else {


            /* generators GENERATE audio */
        
            std::vector<std::vector<void*>> sample_listen_inputs;
            for (double& value : sample_listen) {
                sample_listen_inputs.push_back (std::vector<void*> {static_cast<void*> (&value)});
            }

            // Run the networks over the listening part e.g. it don't predict anything, it just listen for the audio.
            generators.run (sample_listen_inputs, nullptr, MAX_THREADS);

            // Run the networks over the predicting part e.g. it predict the audio.
            std::vector<std::vector<void*>> generated_audio;
            generators.run (N_CALLS_GUESS, &generated_audio, MAX_THREADS);

            // keep in memory the locked genomes
            std::vector<bool> locked_generators (popSize_gen, false);
            for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& generator : generators) {
                locked_generators [generator.first] = generator.second->isLocked ();
            }

            // reset genomes's memory
            generators.resetMemory ();


            /* the discriminator GRADE the generated audio */

            for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& generator : generators) {

                if (!locked_generators [generator.first]) {

                    // run the discriminator
                    for (const double& input : sample_listen) {
                        bestDiscriminator->loadInput (input, 0);
                        bestDiscriminator->runNetwork ();
                    }
                    for (const void* input : generated_audio [generator.first]) {
                        bestDiscriminator->loadInput (input, 0);
                        bestDiscriminator->runNetwork ();
                    }

                    if (!bestDiscriminator->isLocked ()) {
                        const double score = (1.0 - bestDiscriminator->template getOutput<double> (0));
                        // grade the generator
                        if (score > 0.0) {
                            generator.second->setFitness (1.0 / score);
                        } else {
                            generator.second->setFitness (DBL_MAX);
                        }
                    } else {
                        // discriminator diverged, we consider that the generator passed.
                        generator.second->setFitness (DBL_MAX);
                    }

                    // reset the discriminator's memory
                    bestDiscriminator->resetMemory ();

                } else {
                    // the generator raise a NaN value, it already have the worst fitness, a null one
                }

            }


            /* SPECIATE the generators */
            generators.speciate (5, 100, 0.3);


            /* SAVE */
            bestGenerator = generators.getGenome (-1).clone ();
            generators.save ("save/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_gen");


            /* generate the NEW GENERATION of generators */
            generators.crossover (true, 0.7);


            /* MUTATE the generators */
            generators.mutate (paramsMap_gen);


            /* CHECK if the discriminator is still better than the generators */
            if (bestGenerator->getFitness () > 2.0) {
                generator_is_winning = true;
            }

            /* LOG */
            logger->info ("gen_loss {}", 1.0 / bestGenerator->getFitness ());


        }
    }
    return 0;
}