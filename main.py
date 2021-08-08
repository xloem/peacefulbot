print('importing python modules ...')
import torch
import transformers

from time import time
import models
import feelings_needs

# a thought:
#   an organism that uses pretrained machine learning models
#   could be a little similar to an influence spirit,
#      living on the life results of its ancient value actions,
#      in that learning new information is different,
#      and the local environment holding the life of the spirit, may not be living its values
#          - one of the parts is that all values are present everywhere

#textgen = transformers.pipeline('text-generation')
#mark = time()
#print('loading conversational model ...')
# 51s-150s: facebook/blenderbot_small-90M
# 372s?: microsoft/DialoGPT-small
#       facebook/blenderbot-1B-distill
#converse = transformers.pipeline('conversational', model='facebook/blenderbot_small-90M')
converse = models.converse_blenderbot_small_90M
#print((time() - mark), 'seconds')


def options_converse(pipeline, conversations, clean_up_tokenization_spaces=True, **generate_kwargs):
    if isinstance(conversations, transformers.Conversation):
        conversations = [conversations]

    with pipeline.device_placement():
        inputs = pipeline._parse_and_tokenize(conversations)

        if pipeline.framework == 'pt':
            inputs = pipeline.ensure_tensor_on_device(**inputs)
            input_length = inputs['input_ids'].shape[-1]
        generated_responses = pipeline.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **generate_kwargs,
        )
        if pipeline.model.config.is_encoder_decoder:
            if pipeline.framework == 'pt':
                num_responses = generated_responses.shape[0]
                history = torch.cat((inputs['input_ids'].expand(num_responses, -1), generated_responses[:, 1:]), 1)
        else:
            history = generated_responses
        history = pipeline._clean_padding_history(history)
        if pipeline.model.config.is_encoder_decoder:
            start_position = 1
        else:
            start_position = input_length

        num_responses = generated_responses.shape[0] // len(conversations)

        output = []
        for conversation_index, conversation in enumerate(conversations):
            conversation.mark_processed()
            conversation.generated_responses.append(
                [
                    pipeline.tokenizer.decode(
                        generated_responses[response_index][start_position:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                    for response_index in range(conversation_index*num_responses,
                                                (conversation_index + 1) * num_responses)
                ]
            )
            output.append(conversation)
        if len(output) == 1:
            return output[0]
        else:
            return output


feelings = {
    'peaceful': 1,
    'joyful': .5,
    'powerful': 0,
    'sad': 0,
    'scared': 0,
    'mad': -1,
}

conversation = transformers.Conversation()
#conversation.past_user_inputs #can be prefilled
#conversation.generated_responses #can be prefilled
#conversation.past_user_inputs.append('Sup bot.')
#conversation.generated_responses.append('Greetings.  I am Peacefulbot.  I am always yearning and working to learn peaceful, caring wisdom.  Please interrupt me if you would ever like to speak with me.')
#conversation.add_user_input('')

# model will call .mark_processed and .append_response
#print('\nPeacefulbot is learning to discuss plans for community healing.\n\n')
print('\nPeacefulbot likely has time for you if you would like to interrupt.\n\n')
print('What do you say?\n\n')
#while True:
for count in range(1):
    conversation.add_user_input(input('  '))
    options_converse(converse, conversation, num_return_sequences=8, num_beams=8, num_beam_groups=8, diversity_penalty=0.5)
    print(conversation.generated_responses[-1])

# generation options:
# bad_word_ids: tokens that are not allowed to be generated
#         generate bad_word_ids with tokenizer(bad_word, add_prefix_space=True).input_ids
# num_return_sequences: number of generated replies
# max_time: seconds
# attention_mask: (batch_size, sequence_length) tensor with % attention per token
# use_cache: set to False to not cache attentions
# prefix_allowed_tokens_fn: lambda batch_id, input_ids: return a list of allowed tokens given previous tokens
# output_hidden_state: True to return hidden states of all layers
# output_scores: True to return prediction scores
# return_dict_in_generate: True to produce an output object instead of a tuple

