#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <cocos2d.h> // Explicitly include Cocos2d-x headers for all classes (CCLayer, CCDrawNode, CCString, etc.)
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <string>

using namespace geode::prelude;

// ==========================================
// 1. EXPANDED NEURAL NETWORK & REINFORCEMENT LEARNING
// ==========================================

struct FeedForwardResult {
    std::vector<float> inputs;                     // Input Layer (12 values)
    std::vector<std::vector<float>> hidden_layers; // Hidden Layers (Layer 1: 16 values, Layer 2: 8 values)
    float output;                                  // Output Layer (1 value)
};

class NeuralNetwork {
public:
    std::vector<int> topology; // Typically {12, 16, 8, 1}
    std::vector<std::vector<std::vector<float>>> weights; // weights[layer][from_neuron][to_neuron]
    std::vector<std::vector<float>> biases; // biases[layer][neuron]

    NeuralNetwork() {}

    NeuralNetwork(const std::vector<int>& topo) {
        topology = topo;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (size_t i = 0; i < topology.size() - 1; ++i) {
            int num_inputs = topology[i];
            int num_outputs = topology[i + 1];

            std::vector<std::vector<float>> layer_weights;
            std::vector<float> layer_biases;

            for (int j = 0; j < num_inputs; ++j) {
                std::vector<float> neuron_weights;
                for (int k = 0; k < num_outputs; ++k) {
                    neuron_weights.push_back(dis(gen));
                }
                layer_weights.push_back(neuron_weights);
            }

            for (int k = 0; k < num_outputs; ++k) {
                layer_biases.push_back(dis(gen));
            }

            weights.push_back(layer_weights);
            biases.push_back(layer_biases);
        }
    }

    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }

    FeedForwardResult feedForwardDetailed(const std::vector<float>& inputs) const {
        FeedForwardResult res;
        res.inputs = inputs;

        std::vector<float> current_vals = inputs;

        // Propagate forward through hidden layers
        for (size_t i = 0; i < weights.size() - 1; ++i) {
            std::vector<float> next_vals(topology[i + 1], 0.0f);

            for (int k = 0; k < topology[i + 1]; ++k) {
                float sum = biases[i][k];
                for (int j = 0; j < topology[i]; ++j) {
                    sum += current_vals[j] * weights[i][j][k];
                }
                next_vals[k] = sigmoid(sum);
            }
            res.hidden_layers.push_back(next_vals);
            current_vals = next_vals;
        }

        // Final transition to the single Output layer (JUMP / CLICK)
        size_t last_idx = weights.size() - 1;
        float sum_out = biases[last_idx][0];
        for (int j = 0; j < topology[last_idx]; ++j) {
            sum_out += current_vals[j] * weights[last_idx][j][0];
        }
        res.output = sigmoid(sum_out);

        return res;
    }

    void mutate(float rate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, 0.25f);
        std::uniform_real_distribution<float> rand_dis(0.0f, 1.0f);

        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                for (size_t k = 0; k < weights[i][j].size(); ++k) {
                    if (rand_dis(gen) < rate) {
                        weights[i][j][k] += dis(gen);
                        weights[i][j][k] = std::clamp(weights[i][j][k], -2.0f, 2.0f);
                    }
                }
            }

            for (size_t k = 0; k < biases[i].size(); ++k) {
                if (rand_dis(gen) < rate) {
                    biases[i][k] += dis(gen);
                    biases[i][k] = std::clamp(biases[i][k], -2.0f, 2.0f);
                }
            }
        }
    }
};

struct BrainInstance {
    NeuralNetwork network;
    float fitness = 0.0f;
    int index = 0;
};

class GeneticPopulation {
public:
    std::vector<BrainInstance> population;
    int current_generation = 1;
    int current_brain_idx = 0;
    float best_fitness_all_time = 0.0f;
    NeuralNetwork best_network_all_time;
    bool has_best_ever = false;

    GeneticPopulation() {}

    GeneticPopulation(int size, const std::vector<int>& topology) {
        population.resize(size);
        for (int i = 0; i < size; ++i) {
            population[i].network = NeuralNetwork(topology);
            population[i].fitness = 0.0f;
            population[i].index = i;
        }
    }

    BrainInstance& getCurrentBrain() {
        return population[current_brain_idx];
    }

    void setFitness(float fit) {
        population[current_brain_idx].fitness = fit;
        if (fit > best_fitness_all_time) {
            best_fitness_all_time = fit;
            best_network_all_time = population[current_brain_idx].network;
            has_best_ever = true;
        }
    }

    void evolve(float mutation_rate) {
        int best_idx = 0;
        float max_fitness = -1.0f;
        for (size_t i = 0; i < population.size(); ++i) {
            if (population[i].fitness > max_fitness) {
                max_fitness = population[i].fitness;
                best_idx = i;
            }
        }

        NeuralNetwork best_nn = population[best_idx].network;

        population[0].network = best_nn;
        population[0].fitness = 0.0f;

        if (has_best_ever) {
            population[1].network = best_network_all_time;
            population[1].fitness = 0.0f;
        }

        size_t start_idx = has_best_ever ? 2 : 1;
        for (size_t i = start_idx; i < population.size(); ++i) {
            population[i].network = best_nn;
            float rate = mutation_rate + 0.18f * (static_cast<float>(i) / population.size());
            population[i].network.mutate(rate);
            population[i].fitness = 0.0f;
        }

        current_generation++;
        current_brain_idx = 0;
    }

    bool nextBrain(float mutation_rate) {
        current_brain_idx++;
        if (current_brain_idx >= population.size()) {
            evolve(mutation_rate);
            return true;
        }
        return false;
    }
};

// ==========================================
// 2. REAL-TIME MULTI-LAYER NEURAL GRAPH OVERLAY
// ==========================================

class AIOverlay : public cocos2d::CCLayer {
public:
    cocos2d::CCLabelBMFont* m_generation_label;
    cocos2d::CCLabelBMFont* m_brain_label;
    cocos2d::CCLabelBMFont* m_fitness_label;
    cocos2d::CCLabelBMFont* m_best_fitness_label;
    cocos2d::CCDrawNode* m_draw_node;
    cocos2d::CCLayerColor* m_bg_panel;

    static AIOverlay* create() {
        auto* ret = new AIOverlay();
        if (ret && ret->init()) {
            ret->autorelease();
            return ret;
        }
        CC_SAFE_DELETE(ret);
        return nullptr;
    }

    bool init() override {
        if (!CCLayer::init()) return false;

        auto win_size = cocos2d::CCDirector::sharedDirector()->getWinSize();

        m_bg_panel = cocos2d::CCLayerColor::create(cocos2d::ccc4(0, 0, 0, 160), 190, 120);
        m_bg_panel->setPosition(cocos2d::ccp(10, win_size.height - 130));
        this->addChild(m_bg_panel);

        m_generation_label = cocos2d::CCLabelBMFont::create("Generation: 1", "goldFont.fnt");
        m_brain_label = cocos2d::CCLabelBMFont::create("Brain: 1/20", "goldFont.fnt");
        m_best_fitness_label = cocos2d::CCLabelBMFont::create("Best Dist: 0.0", "goldFont.fnt");
        m_fitness_label = cocos2d::CCLabelBMFont::create("Current Dist: 0.0", "chatFont.fnt");

        m_generation_label->setScale(0.55f);
        m_brain_label->setScale(0.50f);
        m_best_fitness_label->setScale(0.50f);
        m_fitness_label->setScale(0.60f);

        m_generation_label->setAnchorPoint({0, 0.5f});
        m_brain_label->setAnchorPoint({0, 0.5f});
        m_best_fitness_label->setAnchorPoint({0, 0.5f});
        m_fitness_label->setAnchorPoint({0, 0.5f});

        m_generation_label->setPosition(cocos2d::ccp(15, 100));
        m_brain_label->setPosition(cocos2d::ccp(15, 80));
        m_best_fitness_label->setPosition(cocos2d::ccp(15, 60));
        m_fitness_label->setPosition(cocos2d::ccp(15, 25));

        m_bg_panel->addChild(m_generation_label);
        m_bg_panel->addChild(m_brain_label);
        m_bg_panel->addChild(m_best_fitness_label);
        m_bg_panel->addChild(m_fitness_label);

        m_draw_node = cocos2d::CCDrawNode::create();
        m_draw_node->setPosition(cocos2d::ccp(win_size.width - 260, 20));
        this->addChild(m_draw_node);

        auto* panel_outline = cocos2d::CCDrawNode::create();
        panel_outline->setPosition(m_draw_node->getPosition());
        panel_outline->drawRect(cocos2d::ccp(0, 0), cocos2d::ccp(245, 160), cocos2d::ccc4f(0.08f, 0.08f, 0.08f, 0.7f), 1.0f, cocos2d::ccc4f(0.6f, 0.6f, 0.6f, 0.9f));
        this->addChild(panel_outline, -1);

        auto* visualizer_title = cocos2d::CCLabelBMFont::create("EXPANDED BRAIN MONITOR (12-16-8-1)", "chatFont.fnt");
        visualizer_title->setScale(0.42f);
        visualizer_title->setOpacity(220);
        visualizer_title->setPosition(cocos2d::ccp(win_size.width - 137, 167));
        this->addChild(visualizer_title);

        return true;
    }

    void drawNeuralNetworkGraph(const NeuralNetwork& nn, const FeedForwardResult& res) {
        m_draw_node->clear();

        size_t num_layers = nn.topology.size();
        if (num_layers < 2) return;

        float panel_width = 245.0f;
        float panel_height = 160.0f;
        float left_margin = 15.0f;
        float right_margin = panel_width - 15.0f;
        float col_spacing = (right_margin - left_margin) / (num_layers - 1);
        float neuron_radius = 4.0f;

        std::vector<std::vector<cocos2d::CCPoint>> layer_pts(num_layers);

        for (size_t l = 0; l < num_layers; ++l) {
            int num_nodes = nn.topology[l];
            float x_pos = left_margin + l * col_spacing;

            float max_usable_height = 135.0f;
            float row_spacing = 15.0f;
            if (num_nodes > 1) {
                row_spacing = std::min(max_usable_height / (num_nodes - 1), 16.0f);
            }

            float y_center = panel_height / 2.0f;
            float y_start = y_center - ((num_nodes - 1) * row_spacing) / 2.0f;

            layer_pts[l].resize(num_nodes);
            for (int i = 0; i < num_nodes; ++i) {
                layer_pts[l][i] = cocos2d::ccp(x_pos, y_start + i * row_spacing);
            }
        }

        // Connections (Weights)
        for (size_t l = 0; l < num_layers - 1; ++l) {
            int num_from = nn.topology[l];
            int num_to = nn.topology[l + 1];

            for (int i = 0; i < num_from; ++i) {
                for (int j = 0; j < num_to; ++j) {
                    float w = nn.weights[l][i][j];
                    cocos2d::ccColor4F color = (w > 0.0f) ? 
                        cocos2d::ccc4f(0.0f, 1.0f, 0.0f, std::abs(w) * 0.45f) : 
                        cocos2d::ccc4f(1.0f, 0.0f, 0.0f, std::abs(w) * 0.45f);
                    m_draw_node->drawSegment(layer_pts[l][i], layer_pts[l + 1][j], 0.5f, color);
                }
            }
        }

        // Nodes (Neurons) - Using drawDot which is the correct Cocos2d-x 2.1 solid circle drawing function!
        for (size_t l = 0; l < num_layers; ++l) {
            int num_nodes = nn.topology[l];

            for (int i = 0; i < num_nodes; ++i) {
                float act = 0.0f;
                if (l == 0) {
                    act = res.inputs[i];
                } else if (l == num_layers - 1) {
                    act = res.output;
                } else {
                    act = res.hidden_layers[l - 1][i];
                }

                cocos2d::ccColor4F fill_color;
                if (l == num_layers - 1) {
                    fill_color = (act > 0.5f) ? 
                        cocos2d::ccc4f(1.0f, 0.85f, 0.0f, 1.0f) : 
                        cocos2d::ccc4f(0.12f, 0.12f, 0.12f, 1.0f);
                } else {
                    float bounded_act = std::clamp(act, 0.0f, 1.0f);
                    fill_color = cocos2d::ccc4f(bounded_act, bounded_act, bounded_act, 1.0f);
                }

                // Cocos2d-x 2.1 trick: draw a slightly larger dot for the outline first, then the actual dot
                m_draw_node->drawDot(layer_pts[l][i], neuron_radius + 0.5f, cocos2d::ccc4f(0.5f, 0.5f, 0.5f, 0.8f));
                m_draw_node->drawDot(layer_pts[l][i], neuron_radius, fill_color);
            }
        }
    }

    void updateOverlay(int gen, int brain, int pop_size, float cur_fit, float best_fit, const NeuralNetwork& nn, const FeedForwardResult& res) {
        // Use 100% compile-safe Cocos2d string formatting (avoids any compiler/STL fmt mismatch errors)
        m_generation_label->setString(cocos2d::CCString::createWithFormat("Generation: %d", gen)->getCString());
        m_brain_label->setString(cocos2d::CCString::createWithFormat("Brain: %d/%d", brain, pop_size)->getCString());
        m_best_fitness_label->setString(cocos2d::CCString::createWithFormat("Best Dist: %.1f", best_fit)->getCString());
        m_fitness_label->setString(cocos2d::CCString::createWithFormat("Current Dist: %.1f", cur_fit)->getCString());

        drawNeuralNetworkGraph(nn, res);
    }
};

// ==========================================
// 3. EFFICIENT SINGLE-PASS ENV SENSOR SCANNER
// ==========================================

struct ObstacleInfo {
    float x_dist = 450.0f;
    float y_diff = 0.0f;
    bool found = false;
};

struct MultiObstacleInfo {
    ObstacleInfo hazard;   
    ObstacleInfo solid;    
    ObstacleInfo ring_pad; 
};

MultiObstacleInfo scanLevelSensors(PlayLayer* layer) {
    MultiObstacleInfo info;
    auto* player = layer->m_player1;
    if (!player) return info;

    float player_x = player->getPositionX();
    float player_y = player->getPositionY();
    float scan_limit = 450.0f;

    float min_hazard_dist = scan_limit;
    float min_solid_dist = scan_limit;
    float min_ring_dist = scan_limit;

    auto* objects = layer->m_objects;
    if (!objects) return info;

    // Fixed loop for Geode v5: use Geode's required CCArrayExt range-based loop!
    for (auto* game_obj : geode::cocos::CCArrayExt<GameObject*>(objects)) {
        if (!game_obj) continue;

        float obj_x = game_obj->getPositionX();
        if (obj_x <= player_x || obj_x > player_x + scan_limit) continue;

        // Compile-safe bindings: Check m_objectType (enum) or specific object IDs as robust fallback definitions
        bool is_hazard = (game_obj->m_objectType == GameObjectType::Hazard) || 
                         (game_obj->m_objectID == 8 || game_obj->m_objectID == 9 || game_obj->m_objectID == 24 || 
                          game_obj->m_objectID == 39 || game_obj->m_objectID == 103 || game_obj->m_objectID == 110 || 
                          game_obj->m_objectID == 183);
                          
        bool is_solid = (game_obj->m_objectType == GameObjectType::Solid) || 
                        (game_obj->m_objectID == 1 || game_obj->m_objectID == 2 || game_obj->m_objectID == 3 || 
                         game_obj->m_objectID == 4);
                         
        // Checked via 100% reliable physical ID classification to avoid non-existent general Ring/Pad enums!
        bool is_ring_pad = (game_obj->m_objectID == 36 || game_obj->m_objectID == 141 || game_obj->m_objectID == 84 || 
                            game_obj->m_objectID == 1022 || game_obj->m_objectID == 1332 || game_obj->m_objectID == 1333 || 
                            game_obj->m_objectID == 1704 || game_obj->m_objectID == 1751 || game_obj->m_objectID == 35 || 
                            game_obj->m_objectID == 140 || game_obj->m_objectID == 67 || game_obj->m_objectID == 1331);

        float dist = obj_x - player_x;

        if (is_hazard && dist < min_hazard_dist) {
            min_hazard_dist = dist;
            info.hazard.x_dist = dist;
            info.hazard.y_diff = game_obj->getPositionY() - player_y;
            info.hazard.found = true;
        }
        else if (is_solid && dist < min_solid_dist) {
            min_solid_dist = dist;
            info.solid.x_dist = dist;
            info.solid.y_diff = game_obj->getPositionY() - player_y;
            info.solid.found = true;
        }
        else if (is_ring_pad && dist < min_ring_dist) {
            min_ring_dist = dist;
            info.ring_pad.x_dist = dist;
            info.ring_pad.y_diff = game_obj->getPositionY() - player_y;
            info.ring_pad.found = true;
        }
    }
    return info;
}

// ==========================================
// 4. LEVEL HOOKS & GAME LOOP CONTROL
// ==========================================

class $modify(MyPlayLayer, PlayLayer) {
    struct Fields {
        bool m_ai_enabled = true;
        GeneticPopulation m_population;
        AIOverlay* m_overlay = nullptr;
        bool m_already_dead = false;
        bool m_last_jump_input = false;
    };

    bool init(GJGameLevel* level, bool usePracticeMode, bool isPlaytest) {
        if (!PlayLayer::init(level, usePracticeMode, isPlaytest)) {
            return false;
        }

        m_fields->m_ai_enabled = Mod::get()->getSettingValue<bool>("ai-enabled");
        int pop_size = static_cast<int>(Mod::get()->getSettingValue<int64_t>("population-size"));

        m_fields->m_population = GeneticPopulation(pop_size, {12, 16, 8, 1});
        m_fields->m_already_dead = false;
        m_fields->m_last_jump_input = false;

        if (m_fields->m_ai_enabled) {
            m_fields->m_overlay = AIOverlay::create();
            if (m_fields->m_overlay) {
                this->addChild(m_fields->m_overlay, 1000);
            }
        }

        return true;
    }

    void update(float dt) {
        PlayLayer::update(dt);

        if (!m_fields->m_ai_enabled || !m_player1 || m_player1->m_isDead) return;

        MultiObstacleInfo sensors = scanLevelSensors(this);

        std::vector<float> inputs(12, 0.0f);
        inputs[0] = std::clamp(sensors.hazard.x_dist / 450.0f, 0.0f, 1.0f);
        inputs[1] = std::clamp(sensors.hazard.y_diff / 150.0f, -1.0f, 1.0f);

        inputs[2] = std::clamp(sensors.solid.x_dist / 450.0f, 0.0f, 1.0f);
        inputs[3] = std::clamp(sensors.solid.y_diff / 150.0f, -1.0f, 1.0f);

        inputs[4] = std::clamp(sensors.ring_pad.x_dist / 300.0f, 0.0f, 1.0f);
        inputs[5] = std::clamp(sensors.ring_pad.y_diff / 150.0f, -1.0f, 1.0f);

        inputs[6] = std::clamp(static_cast<float>(m_player1->m_yVelocity) / 15.0f, -1.0f, 1.0f);
        
        // Changed from non-existent m_xVelocity to valid m_playerSpeed binding
        inputs[7] = std::clamp(static_cast<float>(m_player1->m_playerSpeed) / 15.0f, 0.0f, 1.0f);

        inputs[8] = m_player1->m_isOnGround ? 1.0f : 0.0f;
        inputs[9] = m_player1->m_isUpsideDown ? 1.0f : 0.0f;

        inputs[10] = (m_player1->m_isShip || m_player1->m_isBird || m_player1->m_isDart || m_player1->m_isSwing) ? 1.0f : 0.0f;
        inputs[11] = (m_player1->m_isBall || m_player1->m_isSpider) ? 1.0f : 0.0f;

        BrainInstance& active_brain = m_fields->m_population.getCurrentBrain();
        FeedForwardResult res = active_brain.network.feedForwardDetailed(inputs);

        bool should_jump = (res.output > 0.5f);
        if (should_jump != m_fields->m_last_jump_input) {
            // Explicitly cast PlayerButton to int to fit modern Geode signatures
            this->handleButton(should_jump, static_cast<int>(PlayerButton::Jump), false);
            m_fields->m_last_jump_input = should_jump;
        }

        if (m_fields->m_overlay) {
            float cur_fit = m_player1->getPositionX();
            m_fields->m_overlay->updateOverlay(
                m_fields->m_population.current_generation,
                m_fields->m_population.current_brain_idx + 1,
                m_fields->m_population.population.size(),
                cur_fit,
                m_fields->m_population.best_fitness_all_time,
                active_brain.network,
                res
            );
        }
    }

    void destroyPlayer(PlayerObject* player, GameObject* obstacle) {
        if (m_fields->m_ai_enabled) {
            if (m_fields->m_already_dead) return;
            m_fields->m_already_dead = true;

            float fitness = player->getPositionX();
            m_fields->m_population.setFitness(fitness);

            log::info("Brain died.");

            float base_mut_rate = static_cast<float>(Mod::get()->getSettingValue<double>("mutation-rate"));
            bool evolved = m_fields->m_population.nextBrain(base_mut_rate);
            if (evolved) {
                log::info("Evolved new generation.");
            }

            m_fields->m_last_jump_input = false;
            m_fields->m_already_dead = false;

            this->resetLevel();
            return;
        }

        PlayLayer::destroyPlayer(player, obstacle);
    }

    void levelComplete() {
        if (m_fields->m_ai_enabled) {
            float fitness = m_player1->getPositionX() + 100000.0f;
            m_fields->m_population.setFitness(fitness);
            log::info("Level completed!");

            float base_mut_rate = static_cast<float>(Mod::get()->getSettingValue<double>("mutation-rate"));
            m_fields->m_population.nextBrain(base_mut_rate);
        }

        PlayLayer::levelComplete();
    }
};
