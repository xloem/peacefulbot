feeling_words_by_category = {
    'powerful': 'proud cheerful respected satisfied appreciated valuable hopeful worthwhile important intelligent faithful confident'.split(' '),
    'peaceful': 'nurturing thankful trusting sentimental loving serene intimate responsive thoughtful relaxed content pensive'.split(' '),
    'sad': 'guilty bashful ashamed stupid depressed miserable lonely inadequate bored inferior sleepy apathetic'.split(' '),
    'mad': 'hurt jealous hostile selfish angry frustrated rage furious hateful irritated critical skeptical'.split(' '),
    'scared': 'rejected bewildered confused discouraged helpless insignificant submissive insecure anxious embarrassed'.split(' '),
    # diverse tones. the real meaning of words is what the listener feels when they are said.
    'joyful': 'excited daring sexy fascinating energetic stimulating playful amused creative extravagant aware delightful'.split(' '),
}

# these needs are from a dichotomously-organised nvc needs wheel.
needs_structured = {
   'life': {
       'happiness': {
           'inspiration': ('appreciation', 'beauty'),
           'joy': ('humor', 'delight'),
       },
       'gratefulness': {
           'thankfulness': ('celebration', 'gratitude'),
           'remembrance': ('mourning', 'grieving'),
       },
   },
   'mutuality': {
       'collaboration': {
           'contribution': ('care', 'giving', 'nurturance'),
           'cooperation': ('help', 'support', 'participation'),
       },
       'relationship': {
           'closeness': ('intimacy', 'affection'),
           'community': ('inclusion', 'belonging'),
       },
   },
   'worth': {
       'to matter': {
           'acknowledgement': ('appreciation', 'attention'),
           'respect': ('admiration', 'value'),
       },
       'esteem': {
           'achievement': ('success', 'accomplishment'),
           'recognition': ('validation', 'affirmation'),
       },
   },
   'identity': {
       'uniqueness': {
           'expression': ('creativity', 'originality'),
           'extraordinariness': ('to be known', 'specialness'),
       },
       'authenticity': {
           'genuineness': ('congruence', 'sincerity'),
           'significance': ('purpose', 'meaning'),
       },
   },
   'love': {
       'empathy': {
           'understanding': ('presence', 'patience'),
           'compassion': ('kindness', 'consideration'),
       },
       'connection': {
           'openness': ('tolerance', 'acceptance'),
           'relatedness': ('receptivity', 'sincerity'),
       },
   },
   'self-efficiency': {
       'awareness': {
           'wisdom': ('clarity', 'curiosity'),
           'knowledge': ('learning', 'growth'),
       },
       'agency': {
           'capability': ('skill', 'competence'),
           'confidence': ('effectiveness', 'proficiency'),
       },
   },
   'trust': {
           'assurance': {
               'dependability': ('commitment', 'loyalty'),
               'predictability': ('consistency', 'reliability'),
           },
           'safety': {
               'stability': ('structure', 'order'),
               'protection': ('preparedness', 'security'),
           },
   },
   'fulfillment': {
       'experience': {
           'adventure': ('discovery', 'exploration'),
           'stimulation': ('opportunity', 'challenge'),
       },
       'aliveness': {
           'richness': ('variety', 'excitement'),
           'enjoyment': ('spontaneity', 'fun', 'play'),
       },
   },
   'well-being': {
       'sustenance': {
           'vitality': ('liveliness', 'vivacity'),
           'energy': ('nourishment', 'nutrition'),
       },
       'health': {
           'self-care': ('comfort', 'relief'),
           'rejuvenation': ('rest', 'relaxation', 'healing'),
       },
   },
   'autonomy': {
       'independence': {
           'freedom': ('choice', 'self-determination'),
           'separateness': ('space', 'privacy', 'boundaries'),
       },
       'individuality': {
           'strength': ('courage', 'resolve'),
           'leadership': ('decisiveness', 'influence'),
       },
   },
   'harmony': {
       'acceptance': {
           'tolerance': ('modesty', 'humility'),
           'fluidity': ('convenience', 'ease'),
       },
       'peace': {
           'balance': ('serenity', 'calmness'),
           'unity': ('consensus', 'agreement'),
       },
   },
   'wholeness': {
       'integrity': {
           'truthfulness': ('honesty', 'goodness'),
           'fairness': ('equality', 'justice'),
       },
       'responsibility': {
           'accountability': ('right action', 'certitude'),
           'civility': ('courteousness', 'politeness'),
       },
   },
}

feeling_category_by_word = {}
for feeling_name, feeling_words in feeling_words_by_category.items():
    feeling_category_by_word[feeling_name] = feeling_name
    for feeling_word in feeling_words:
        feeling_category_by_word[feeling_word] = feeling_name

feeling_categories = [*feeling_words_by_category.keys()]
feeling_words = [*feeling_category_by_word.keys()]

def classify_feelings(pipeline, phrases, multi_label = False):
    results = pipeline(phrases, feeling_words, 'This person is feeling {}.', multi_label = multi_label)
    for idx, result in enumerate(results):
        scores = {label: [0.0, None, 0.0] for label in feeling_categories}
        for label, score in zip(result['labels'], result['scores']):
            score_obj = scores[feeling_category_by_word[label]]
            score_obj[0] += score
            if score_obj[2] < score:
                score_obj[1] = label
                score_obj[2] = score
        result = [(label, *score) for label, score in scores.items()]
        result.sort(key = lambda pair: pair[1], reverse = True)
        results[idx] = result
    return results


