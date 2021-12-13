#!/bin/env python
import json
import re

# state abbreviations
state_abbreviations = 'Ala\\.|Ariz\\.|Ark\\.|Cal\\.|Colo\\.|Conn\\.|Del\\.|Fla\\.|Ga\\.|Haw\\.|Ill\\.|Ind\\.|Kan\\.|Ky\\.|La\\.|Me\\.|Md\\.|Mass\\.|Mich\\.|Minn\\.|Miss\\.|Mo\\.|Mont\\.|Neb\\.|Nev\\.|N\\.H\\.|N\\.J\\.|N\\.M\\.|N\\.Y\\.|N\\.C\\.|N\\.D\\.|Okla\\.|Or\\.|Pa\\.|R\\.I\\.|S\\.C\\.|S\\.D\\.|Tenn\\.|Tex\\.|Vt\\.|Va\\.|Wash\\.|Wis\\.|Wyo\\.'

state_abbreviations_dict = {'Ala.': 'Alabama', 'Ariz.': 'Arizona', 'Ark.': 'Arkansas', 'Cal.': 'California',
                            'Colo.': 'Colorado', 'Conn.': 'Connecticut', 'Del.': 'Delaware', 'Fla.': 'Florida',
                            'Ga.': 'Georgia', 'Haw.': 'Hawaii', 'Ill.': 'Illinois', 'Ind.': 'Indiana', 'Kan.': 'Kansas',
                            'Ky.': 'Kentucky', 'La.': 'Louisiana', 'Me.': 'Maine', 'Md.': 'Maryland',
                            'Mass.': 'Massachusetts', 'Mich.': 'Michigan', 'Minn.': 'Minnesota', 'Miss.': 'Mississippi',
                            'Mo.': 'Missouri', 'Mont.': 'Montana', 'Neb.': 'Nebraska', 'Nev.': 'Nevada',
                            'N.H.': 'New Hampshire', 'N.J.': 'New Jersey', 'N.M.': 'New Mexico', 'N.Y.': 'New York',
                            'N.C.': 'North Carolina', 'N.D.': 'North Dakota', 'Okla.': 'Oklahoma', 'Or.': 'Oregon',
                            'Pa.': 'Pennsylvania', 'R.I.': 'Rhode Island', 'S.C.': 'South Carolina',
                            'S.D.': 'South Dakota', 'Tenn.': 'Tennessee', 'Tex.': 'Texas', 'Vt.': 'Vermont',
                            'Va.': 'Virginia', 'Wash.': 'Washington', 'Wis.': 'Wisconsin', 'Wyo.': 'Wyoming'}

# regex pattersn

# regular expressions and replacement

# byte definitions

emdash1_reg = b'\xe2\x80\x94'.decode('utf-8') # emdash
emdash2_reg = b'\xe2\x80\x93'.decode('utf-8') # emdash
emsp_reg = b'\xe2\x80\x82'.decode('utf-8') # space
ensp_reg = b'\xe2\x80\x83'.decode('utf-8') # space
pg = b'\xc2\xb6'.decode('utf-8')


# smart single quote
single_quote_reg1 = b'\xe2\x80\x99'.decode('utf-8') # single smart quote
single_quote_reg2 = b'\xe2\x80\x98'.decode('utf-8') # single smart quote

#smart double quotes
left_smart_quote = b'\xe2\x80\x9c'.decode('utf-8')
right_smart_quote = b'\xe2\x80\x9d'.decode('utf-8')

lpat = re.compile(left_smart_quote)
rpat = re.compile(right_smart_quote)
quote_pattern = re.compile(f'({left_smart_quote}.+?{right_smart_quote})')

chapter_reg = '([cC]hapter\s+?(7|9|11|12|13|15))'
embedded_date_reg = '\d{1,2}/\d{1,2}/\d{2,4}'
empty_group_reg = '[\(\[]\s*[\)\]]' #[]
filled_bracket_reg = '\[(\s*\w.+?\s*?)\]' #[.+]
number_ident_reg = 'no*\.\s*?\d+'
usc_reg = f'U\.\s*S\.\s*C\.'
usc_citation_reg = '(\d*\s+U\.\s+S\.\s+C\.\s+.+?)\s'
us_reg = f'U\.\s+\S\.'
unbound_punctuation_reg = '\s[^\w\s]\s'
unbound_character_reg = '\s[b-hjzB-HJ-Z]\s'
roman_numeral_reg = ' ((M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V))|XI{1,3}|VI{1,3}|I{2,3})\s'


hyphen_reg = f'([a-z]{2})({emdash2_reg})([a-z]{2})'

#####################################################
# case reference regexes
#USC_case_ref_reg = '(\d+\s+U\.\s+S\.\s+C\.\s+?.+?.+?\))[\.;,]' # e.g, 42 U. S. C. §1395dd; Fla. Statute §395.1041(3)(f) (2010)
#versus_case_ref_reg = '[A-Z][a-z]+\.?(?=\s[A-Z])(?:\s[A-Za-z]+\.?)+\s+v\.\s+.+?\([0-9]{4}\)' # eg, North American of Co. v. SEC, 327 U. S. 686, 705 (1946)
#APC_reg = 'App\. to Pet\. for Cert\.' # -> 'appendix to petition for writ of certiorari'

#USC_case_ref_reg = '(\d+\s+U\.\s+S\.\s+C\.\s+?(?:(§\d+|.+?.+?\)|\(.+?Supp.+?\))(?=[\.;, ])))'

#match = re.findall('(\d+\s+U\.\s+S\.\s+C\.\s+?.+?.+?\))[\.;,]', new_content)

#######################################


admin_reg = '([^\w])([Aa])dmin\.'
adsn_reg = '([^\w])aff d sub nom'
am_u_lr_reg = '([^\w])Am\.\s+U\.\s+L\.\s+Rev\.'
ann_reg = '([^\w])([Aa])nn\.'
apc_reg = '([^\w])App\.\s+?to\s+?Pet\.\s+?for\s+?Cert\.' # 'App. to Pet. for Cert.' -> 'appendix to petition for writ of certiorari'
app_reg = '([^\w])App\.'
appx_reg  = '([^\w])([Aa])ppx\.'
ap_reg = '([^\w])App\.\s+?to\s+?Pet\.'
art_reg = '([^\w])([Aa])rt\.'
assn_reg = '([^\w])([Aa])ssn\.'
assocs_reg = '([^\w])Assocs\.'
attach_reg = '([^\w])([Aa])ttach\.'
atty_reg = r'([^\w])([Aa])tty\.'
bkrtcy_reg = '([^\w])([Bb])krtcy\.'
bus_org_reg = '([^\w])([Bb])us\.\s?([Oo])rgs\.'
cert_reg = '([^\w])([Cc])ert\.'
cf_reg = '([^\w])([Cc])f\.'
cfr_pts_reg = '([^\w])CFR\s?pts\.'
chi_reg = '([^\w])Chi\.'
ch_reg = '([^\w])([Cc])h\.'
civ_proc_reg = '([^\w])Civ\.\s+?Proc\.'
civ_reg = '([^\w])Civ\.'
cl_reg = '([^\w])(§\d+,?)\s?([Cc])l\.'
colum_bus_lr_reg = '([^\w])Colum\.\s+Bus\.s+L\.\s+Rev\.'
colum_lr_reg = '([^\w])Colum\.\s+L\.\s+Rev\.'
comm_print_reg = '([^\w])Comm\. Print'
comm_reg = '([^\w])Comm\.'
comp_reg = '([^\w])Comp\.'
com_reg = '([^\w])Com\.'
cong_rec_reg = '([^\w])Cong\.\s+Rec\.'
cong_reg = '([^\w])Cong\.'
cons_reg = '([^\w])Cons\.'
const_reg = '([^\w])Const\.'
constr_reg = '([^\w])Constr\.'
contl_reg = r"([^\w])Cont'l\."
correc_reg = '([^\w])Correc\.'
cpl_reg = '([^\w])Cpl\.'
crim_reg = '([^\w])([Cc])rim\.'
ct_cl_reg = '([^\w])Ct\.\s?Cl\.'
cum_reg = '([^\w])([Cc])um\.'
decl_reg = '([^\w])([Dd])ecl\.'
dep_reg = '([^\w])([Dd])ep\.'
dept_reg = '([^\w])([Dd])ept\.'
digit_stat_reg = '(\d+)Stat\.'
dist_reg = '([^\w])Dist\.'
div_reg = '([^\w])([Dd])iv\.'
ed_reg = '([^\w])([Ee])d\.'
educ_reg = '([^\w])Educ\.'
elipsis_reg = r'\.{3}'
empl_reg = r'([^\w])Empl\.'
eng_rep_reg = '([^\w])Eng\.\s?Rep\.'
esp_reg = '([^\w])([Ee])sp\.'
et_al_reg = '([^\w])et al\.'  # I really hate having to do this one.
et_seq_reg = '([^\w])([Ee])t\.\s?([Ss])eq\.'
eur_ct_reg = '([^\w])Eur\.\s?Ct\.'
evid_reg = '([^\w])([Ee])vid\.'
exec_reg = '([^\w])([Ee])xec\.'
exh_pat = '([^\w])([Ee])xh\.'
exhs_pat = '([^\w])([Ee])xhs\.'
ex_rel_reg = '([^\w])ex rel\.'
fam_reg = '([^\w])Fam\.'
fed_cas_reg = '([^\w])Federal\s?Cas\.'
fed_reg = '([^\w])([Ff])ed\.'
footnote_pat = '\[?\s?[Ff]ootnote\s?\d*\s?\]?'
f_reg = '([^\w])F\.'
fsup_reg = '([^\w])F\.\s+Supp\.'
geo_reg = '([^\w])Geo\.'
govt_reg = '([^\w])Govt\.'
harv_lr_reg = '([^\w])Harv\.s+L\.\s+Rev\.'
harv_reg = '([^\w])Harv\.'
h_res_reg = '([^\w])H\.\s?Res\.'
hum_beh_reg = '([^\w])Hum\. Behav\.'
hwy_reg = '([^\w])hwy\.'
indus_reg = '([^\w])Indus\.'
infra_reg = '([^\w])([Ii])nfra\.'
inj_reg = '([^\w])([Ii])nj\.'
invs_reg = '([^\w])([Ii])nvs\.'
jdgmt_reg = '([^\w])([Jj])dgmt\.'
jlpp_reg = "([^\w])J\.\s+L\.\s+\&\s+Pub\.\s+Pol'y"
jlp_reg = "([^\w])J\.\s+L\.\s+\&\s+Pol'y"
justice_reg = '([^\w])\,\s?J\.'
lcn_reg = '([A-Za-z])\,(\d)'
legal_soc_reg = '([^\w])Legal Soc\.'
lower_rev_reg = '([^\w])rev\.'
l_rev_reg = '([^\w])L\.\s+Rev\.'
mfrs_reg = '([^\w])Mfrs\.'
mun_reg = '([^\w])([Mm])un\.'
nat_reg = '([^\w])Nat\.'
nd_cent_reg = '([^\w])N\.\s?D\.\s?Cent\.'
no_reg = '([^\w])([Nn])o\.'
nyu_lr_reg = '([^\w])N\.\s+Y\.\s+U\.\s+L\.\s+Rev\.'
op_reg = '([^\w])([Oo])p\.'
pet_reg = '([^\w])([Pp])et\.'
pp_reg = '([^\w])pp\.'
prac_reg = '([^\w])Prac\.'
prelim_reg = '([^\w])([Pp])relim\.'
proc_reg = '([^\w])([Pp])roc\.'
prods_reg = '([^\w])prods\.'
pt_reg = '([^\w])pt\.'
pub_l_reg = '([^\w])Pub\.\s+L\.'
pub_serv_com_reg = "([^\w])Public\s?Serv\.\s?Comm'n"
rcd_reg = r'([^\w])Rcd\.'
reg_reg = '([^\w])([Rr])eg\.'
rept_reg = '([^\w])([Rr])ept\.'
rev_comm_reg = 'Reservists Comm\.'
rev_reg = '([^\w])([Rr])ev\.'
rev_rul_reg = r'([^\w])Rev\.]s?Rul\.'
rptr_reg = '([^\w])Rptr\.'
s_ct_reg = '([^\w])S\.\s?Ct\.'
section_reg = '([^\w])(§\s+?\d+)'
section_dot_reg = '([^\w])(§\d+)\.(\d+)'
servs_reg = '([^\w])([Ss])ervs\.'
sess_reg = '([^\w])([Ss])ess\.'
soc_reg = '([^\w])([Ss])oc\.'
southern_reg = '([^\w])(\d+)\s?So.'
stan_lr_reg = '([^\w])Stan\.\s+L\.\s+Rev\.'
stat_reg = '([^\w])([Ss])tat\.'
sup_ct_reg = '([^\w])Sup\.\s?Ct\.'
sup_reg = '([^\w])([Ss])upp\.'
surr_ct_reg = '([^\w])([Ss])urr\. ([Cc])t\.'
three_abbr_reg = '\s+?([A-Z]{1}\.)\s+?([A-Z]{1}\.)\s+?([A-Z]{1}\.)\s?'
tit_reg = '([^\w])([Tt])it\.'
transp_reg = '([^\w])Transp\.'
trarg_reg = '([^\w])([Tt])r\.\s+of\s+([Oo])ral\s+([Aa])rg\.'
treas_reg = '([^\w])Treas\.'
two_abbr_reg = '\s+?([A-Z]{1}\.)\s+?([A-Z]{1}\.)\s?'
u_chi_lr_reg = '([^\w])U\.\s+Chi\.\s+L\.\s+Rev\.'
univ_reg = '([^\w])Univ\.'
u_pa_lr_reg = '([^\w])U\.\s+Pa\.\s+L\.\s+Rev\.'
upper_rev_reg = '([^\w])Rev\.'
urb_reg = '([^\w])Urb\.'
usem_reg = r'em\>'
usf_lr_reg = '([^\w])U\.\s+S\.\s+F\.\s+L\.\s+Rev\.'
util_reg = '([^\w])([Uu])til\.'
vand_lr_reg = '([^\w])Vand\.\s+L\.\s+Rev\.'
versus_reg = '\s{1}([Vv]{1})\.\s{1}'
v_reg = '([^\w])([Vv]) \.'
wm_lr_reg = '([^\w])Wm\.\s+\&\s+Mary\s+L\.\s+Rev\.'
ws_reg = '\s+'


l_quote_space_reg = f'{left_smart_quote} '
r_quote_space_reg = f' {right_smart_quote}'

three_dot_reg = '\s?(\.\s+\.\s+\.\s+)' # ' . . . ' -> ' '
three_star_reg = '\s?(\*\s+\*\s+\*\s+)' # ' * * * ' -> ' '


def initial_clean(txt):
    matches = re.findall(state_abbreviations, txt)
    for match in matches:
        if match == 'W. Va':
            txt = txt.replace('W. Va', 'West Virginia')
        else:
            txt = txt.replace(match, state_abbreviations_dict[match])
        
    # abbreviation substitutions

    txt = re.sub('\s+', ' ', txt)
    txt = re.sub(single_quote_reg1, "'", txt)
    txt = re.sub(single_quote_reg2, "'", txt)
    txt = re.sub(l_quote_space_reg, left_smart_quote, txt)
    txt = re.sub(r_quote_space_reg, right_smart_quote, txt)
    
    txt = re.sub(apc_reg, r'\1appendix to petition for writ of certiorari', txt) # add to matcher
    txt = re.sub(ap_reg, r'\1appendix to petition', txt) # add to matcher
    txt = re.sub(trarg_reg, r'\1\2ranscript of \3ral \4argument', txt)
    txt = re.sub(civ_proc_reg, r'\1Civil Procedure', txt) # add to matcher
    txt = re.sub(fsup_reg, r'\1Federal Supplement', txt) # add to matcher
    txt = re.sub(f_reg, r'\1Federal', txt)
    
    txt = re.sub(jlpp_reg, r'\1Journal Of Law and Public Policy', txt)
    txt = re.sub(jlp_reg, r'\1Journal Of Law and Policy', txt)
    
    txt = re.sub(rev_rul_reg, r'\1Revenue Rulings', txt)
    txt = re.sub(et_al_reg, r'\1and others', txt)
    txt = re.sub(pp_reg, r'\1pages', txt)
    
    txt = re.sub(harv_lr_reg, r'\1Harvard Law Review', txt)
    txt = re.sub(colum_bus_lr_reg, r'\1Columbia Business Law Review', txt)    
    txt = re.sub(colum_lr_reg, r'\1Columbia Law Review', txt)
    txt = re.sub(u_chi_lr_reg, r'\1University of Chicago Law Review', txt)
    txt = re.sub(vand_lr_reg, r'\1Vanderbilt Law Review', txt)
    txt = re.sub(stan_lr_reg, r'\1Stanford Law Review', txt)
    txt = re.sub(u_pa_lr_reg, r'\1University of Pennsylvania Law Review', txt)
    txt = re.sub(nyu_lr_reg, r'\1New York University Law Review', txt)
    txt = re.sub(am_u_lr_reg, r'\1American University Law Review', txt)
    txt = re.sub(usf_lr_reg, r'\1University of Southern Florida Law Review', txt)
    txt = re.sub(wm_lr_reg, r'\1William and Mary Law Review', txt)
    txt = re.sub(l_rev_reg, r'\1Law Review', txt)
    
    txt = re.sub(harv_reg, r'\1Harvard', txt)

    txt = re.sub(digit_stat_reg, r' \1 Statute ', txt)
    txt = re.sub(educ_reg, r'\1Education', txt)
    txt = re.sub(assn_reg, r'\1\2ssociation', txt)
    txt = re.sub(cong_rec_reg, r'\1Congressional Record', txt)
    txt = re.sub(pub_l_reg, r'\1Public Laws', txt)
    txt = re.sub(urb_reg, r'\1Urban', txt)
    txt = re.sub(ann_reg, r'\1\2nnotation', txt)
    txt = re.sub(dep_reg, r'\1\2eposition', txt)
    txt = re.sub(ex_rel_reg, r'\1on the relation of', txt)
    txt = re.sub(inj_reg, r'\1\2njunction', txt)
    txt = re.sub(prelim_reg, r'\1\2reliminary', txt)
    txt = re.sub(div_reg, r'\1\2ivision', txt)
    txt = re.sub(mun_reg, r'\1\2unicipal', txt)
    txt = re.sub(attach_reg, r'\1\2ttachment', txt)
    txt = re.sub(reg_reg, r'\1\2egulation', txt)
    txt = re.sub(civ_reg, r'\1Civil', txt)
    txt = re.sub(cl_reg, r'\1\2 \3lause', txt)
    txt = re.sub(section_dot_reg, r'\1', txt)
    txt = re.sub(section_reg, r'\1', txt)
    txt = re.sub(cf_reg, r'\1\2ompare', txt)
    txt = re.sub(const_reg, r'\1Constitution', txt)
    txt = re.sub(pet_reg, r'\1\2etition', txt)
    txt = re.sub(cert_reg, r'\1\2ertiorari', txt)
    txt = re.sub(ed_reg, r'\1\2dition', txt)
    txt = re.sub(stat_reg, r'\1\2tatute', txt)
    txt = re.sub(sup_reg, r'\1\2upplement', txt)
    txt = re.sub(art_reg, r'\2rticle', txt)
    txt = re.sub(no_reg, r'\1\2umber', txt)
    txt = re.sub(op_reg, r'\1\2pinion', txt)
    txt = re.sub(dist_reg, r'\1District', txt)
    txt = re.sub(app_reg, r'\1Application', txt)
    txt = re.sub(fed_reg, r'\1\2ederal', txt)
    txt = re.sub(exh_pat, r'\1\2xhibit', txt)
    txt = re.sub(exhs_pat, r'\1\2xhibits', txt)
    txt = re.sub(tit_reg, r'\1\2itle', txt)
    txt = re.sub(cum_reg, r'\1\2ummulative', txt)
    txt = re.sub(crim_reg, r'\1\2riminal', txt)
    txt = re.sub(servs_reg, r'\1\2ervices', txt)
    txt = re.sub(proc_reg, r'\1\2rocedure', txt)
    txt = re.sub(footnote_pat, ' ', txt)
    txt = re.sub('§+', '§', txt)

    txt = re.sub('\s?(\.\s+\.\s+\.\s+)', ' ', txt) # ' . . . ' -> ' '
    txt = re.sub('\s?(\*\s+\*\s+\*\s+)', ' ', txt) # ' * * * ' -> ' '
    txt = re.sub(emdash1_reg, '-', txt)
    txt = re.sub(emdash2_reg, '-', txt)
    txt = re.sub('([^\w])-(\w+?)', r'\1 - \2', txt)
    txt = re.sub(util_reg, r'\1\2tility', txt)
    txt = re.sub(jdgmt_reg, r'\1\2udgment', txt)
    txt = re.sub(rev_reg, r'\1\2eview', txt)
    txt = re.sub(cong_reg, r'\1Congress', txt)
    txt = re.sub(comm_print_reg, r'\1Commercial Print', txt)
    txt = re.sub(univ_reg, r'\1University', txt)
    txt = re.sub(legal_soc_reg, r'\1Legal Society', txt)
    txt = re.sub(stat_reg, r'\1 Statues at Large', txt)
    txt = re.sub(rev_comm_reg, r'Reservists Committee', txt)
    txt = re.sub(admin_reg, r'\1\2dministration', txt)
    txt = re.sub(surr_ct_reg, r'\1\2urrogate \3ourt', txt)
    txt = re.sub(adsn_reg, r'\1affirmed sub nom', txt)
    txt = re.sub(sess_reg, r'\1\2ession', txt)
    txt = re.sub(appx_reg, r'\1\1ppendix', txt)
    txt = re.sub(et_seq_reg, r'\1\2t \3seq', txt)
    txt = re.sub(constr_reg, r'\1Construction', txt)
    txt = re.sub(soc_reg, r'\1\2ociety', txt)
    txt = re.sub(exec_reg, r'\1\2xecutive', txt)
    txt = re.sub(esp_reg, r'\1\2specially', txt)
    txt = re.sub(esp_reg, r'\1\2nvestments', txt)
    txt = re.sub(decl_reg, r'\1\2eclaration', txt)
    txt = re.sub(southern_reg, r'\1\2 Southern Reporter', txt)
    txt = re.sub(nd_cent_reg, r'\1North Dakota Century', txt)
    txt = re.sub(pub_serv_com_reg, r'\1Public Service Commission', txt)
    txt = re.sub(ct_cl_reg, r'\1Court of Federal Claims Reporter', txt)
    txt = re.sub(bkrtcy_reg, r'\1\2ankruptcy', txt)
    txt = re.sub(bus_org_reg, r'\1\2usiness \3rganizations', txt)
    txt = re.sub(eng_rep_reg, r'\1English Reports', txt)
    txt = re.sub(mfrs_reg, r'\1Manufacturers', txt)
    txt = re.sub(hwy_reg, r'\1Highway', txt)
    txt = re.sub(geo_reg, r'\1George', txt)
    txt = re.sub(infra_reg, r'\1\2nfrastructure', txt)
    txt = re.sub(nat_reg, r'\1National', txt)
    txt = re.sub(rptr_reg, r'\1Reporter', txt)
    txt = re.sub(comp_reg, r'\1Compensation', txt)
    txt = re.sub(elipsis_reg, r' ', txt)
    txt = re.sub(pt_reg, r'\1part', txt)
    txt = re.sub(empl_reg, 'r\1Employee', txt)
    txt = re.sub(atty_reg, r'\1\2ttorney', txt)
    txt = re.sub(contl_reg, r'\1Continental', txt)
    txt = re.sub(rcd_reg, r'\1Record', txt)
    txt = re.sub(usem_reg, r' ', txt)
    txt = re.sub(rept_reg, r'\1\2eport', txt)
    txt = re.sub(prac_reg, 'r\1Practice', txt)
    txt = re.sub(justice_reg, r'\1, Justice', txt)
    txt = re.sub(correc_reg, r'\1Corrections', txt)
    txt = re.sub(sup_ct_reg, r'\1Superior Court', txt)
    txt = re.sub(ch_reg, r'\1\2hapter', txt)
    txt = re.sub(lcn_reg, r'\1, \2', txt)
    txt = re.sub(dept_reg, r'\1Department', txt)
    txt = re.sub(transp_reg, r'Transportation', txt)
    txt = re.sub(chi_reg, r'\1Chicago', txt)
    txt = re.sub(s_ct_reg, r'\1Supreme Court Reporter', txt)
    txt = re.sub(lower_rev_reg, r'\1revised', txt)
    txt = re.sub(upper_rev_reg, r'\1Revenue', txt)
    txt = re.sub(cpl_reg, r'\1Corporal', txt)
    txt = re.sub(hum_beh_reg, r'\1Human Behavior', txt)
    txt = re.sub(prods_reg, r'\1Products', txt)
    txt = re.sub(comm_reg, r'\1Committee', txt)
    txt = re.sub(eur_ct_reg, r'\1European Court', txt)
    txt = re.sub(assocs_reg, r'\1Associates', txt)
    txt = re.sub(treas_reg, 'r\1Treasury', txt)
    txt = re.sub(fam_reg, r'\1Family', txt)
    txt = re.sub(indus_reg, r'\1Industry', txt)
    txt = re.sub(fed_cas_reg, r'\1Federal Cost Accounting Standards', txt)
    txt = re.sub(cfr_pts_reg, r'\1CFR Parts', txt)
    txt = re.sub(com_reg, r'\1Common', txt)
    txt = re.sub(govt_reg, r'\1Goverment', txt)
    txt = re.sub(cons_reg, r'\1Constitutional', txt)
    txt = re.sub(h_res_reg, r'\1House Resolution', txt)
    txt = re.sub(evid_reg, r'\1\2vidence', txt)
    txt = re.sub(versus_reg, r' \1ersus ', txt)
    txt = re.sub(three_abbr_reg, r' \1\2\3 ', txt)
    txt = re.sub(two_abbr_reg, r' \1\2 ', txt)

    txt = re.sub(roman_numeral_reg, '', txt)

    txt = txt.replace('\xa0', ' ')
    txt = txt.replace('\x5f', ' ')

    txt = re.sub('\s+', ' ', txt)
    txt = re.sub(l_quote_space_reg, left_smart_quote, txt)
    txt = re.sub(r_quote_space_reg, right_smart_quote, txt)
    txt = re.sub(unbound_character_reg, ' ', txt)
    txt = re.sub(unbound_punctuation_reg, ' ', txt)
    txt = re.sub('\s+', ' ', txt)    
    
    return txt
